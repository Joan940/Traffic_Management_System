import traci
import sys
from constraint import Problem, BacktrackingSolver
import numpy as np
from sumoenv import SumoEnv 

class TrafficLightCSP:
    def __init__(self):
        self.env = SumoEnv(label='csp_sim', gui_f=True)
        self.tl_id = "gneJ00" 
        self.ns_lanes = ['-gneE0_0', '-gneE0_1', '-gneE0_2', '-gneE2_0', '-gneE2_1', '-gneE2_2']
        self.ew_lanes = ['-gneE1_0', '-gneE1_1', '-gneE1_2', '-gneE3_0', '-gneE3_1', '-gneE3_2']
        
        # SET LAMPU LALU LINTAS
        self.min_green = 20
        self.max_green = 60
        self.yellow_time = 5
        self.red_time = 0 

        self.step = 0
        self.total_vehicles_departed = 0
        self.total_waiting_time = 0.0
        self.vehicle_travel_times = {}
        self.vehicle_departure_times = {}
        
        # VARIABEL UNTUK MENAMPILKAN LOG
        self.current_ns_waiting_time = 0.0
        self.current_ew_waiting_time = 0.0
        self.current_ns_queue_length = 0
        self.current_ew_queue_length = 0
        
        # STEP YANG BISA DISESUAIKAN    <================================================================================
        self.max_simulation_steps = 500 

    def _update_vehicle_metrics(self):
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in self.vehicle_departure_times:
                self.vehicle_departure_times[veh_id] = self.step

        # MENGHITUNG KENDARAAN YANG SAMPAI KETUJUAN
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            if veh_id in self.vehicle_departure_times:
                travel_time = self.step - self.vehicle_departure_times[veh_id]
                self.vehicle_travel_times[veh_id] = travel_time
                del self.vehicle_departure_times[veh_id]

    def _get_current_lane_metrics(self):
        ns_queue = 0
        ew_queue = 0
        ns_waiting_sum = 0.0
        ew_waiting_sum = 0.0
        ns_vehicle_count_for_avg_wait = 0
        ew_vehicle_count_for_avg_wait = 0

        for lane in self.ns_lanes:
            ns_queue += traci.lane.getLastStepHaltingNumber(lane)
            current_lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in current_lane_vehicles:
                ns_waiting_sum += traci.vehicle.getWaitingTime(veh_id)
                ns_vehicle_count_for_avg_wait += 1
        
        for lane in self.ew_lanes:
            ew_queue += traci.lane.getLastStepHaltingNumber(lane)
            current_lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in current_lane_vehicles:
                ew_waiting_sum += traci.vehicle.getWaitingTime(veh_id)
                ew_vehicle_count_for_avg_wait += 1

        self.current_ns_waiting_time = (ns_waiting_sum / ns_vehicle_count_for_avg_wait) if ns_vehicle_count_for_avg_wait > 0 else 0.0
        self.current_ew_waiting_time = (ew_waiting_sum / ew_vehicle_count_for_avg_wait) if ew_vehicle_count_for_avg_wait > 0 else 0.0
        
        self.current_ns_queue_length = ns_queue
        self.current_ew_queue_length = ew_queue

    def _run_phase(self, phase_duration, phase_id):
        self.env.set_traffic_light_phase(phase_id, phase_duration)
        for _ in range(int(phase_duration)):

            # BREAK KETIKA STEP MENCAPAI BATAS
            if self.step >= self.max_simulation_steps:
                break

            self.env.simulation_step()
            self._update_vehicle_metrics()
            
            total_halting_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self.ns_lanes + self.ew_lanes)
            current_total_waiting_time_step = self.env.get_waiting_time()
            self.total_waiting_time += current_total_waiting_time_step
            
            with open('queue_length.txt', 'a') as f:
                f.write(f"{self.step},{total_halting_vehicles},{current_total_waiting_time_step},"
                        f"{self.current_ns_queue_length},{self.current_ew_queue_length},"
                        f"{self.current_ns_waiting_time:.2f},{self.current_ew_waiting_time:.2f}\n")
            self.step += 1

    def run(self):
        self.env.reset()

        # MENULIS LOG HASIL PADA FILE
        with open('queue_length.txt', 'w') as f:
            f.write("step,total_halting_vehicles,total_waiting_time_step,ns_queue,ew_queue,ns_avg_wait_current,ew_avg_wait_current\n")

        try:
            while self.step < self.max_simulation_steps: 
                self.total_vehicles_departed = len(self.vehicle_travel_times)
                if self.step >= self.max_simulation_steps:
                    break

                self._get_current_lane_metrics()
                
                ns_demand_metric = self.current_ns_queue_length + (self.current_ns_waiting_time * 5) 
                ew_demand_metric = self.current_ew_queue_length + (self.current_ew_waiting_time * 5) 

                ns_demand_metric = max(ns_demand_metric, 1)
                ew_demand_metric = max(ew_demand_metric, 1)

                total_demand = ns_demand_metric + ew_demand_metric

                ns_ratio = ns_demand_metric / total_demand
                ew_ratio = ew_demand_metric / total_demand

                csp = Problem(BacktrackingSolver(forwardcheck=True))
                
                total_green_budget = 120 - (2 * self.yellow_time) 

                target_green_ns = int(total_green_budget * ns_ratio)
                target_green_ew = int(total_green_budget * ew_ratio)

                target_green_ns = max(self.min_green, min(target_green_ns, self.max_green))
                target_green_ew = max(self.min_green, min(target_green_ew, self.max_green))
                
                # MEMILIH VARIABLE DALAM RANGE DI BAWAH INI
                csp.addVariable('green_ns', range(max(self.min_green, target_green_ns - 10), min(self.max_green, target_green_ns + 10) + 1))
                csp.addVariable('green_ew', range(max(self.min_green, target_green_ew - 10), min(self.max_green, target_green_ew + 10) + 1))

                # DEBUGG BUGGG BUGGG BUGGG
                print(range(max(self.min_green, target_green_ew - 10), min(self.max_green, target_green_ew + 10) + 1))

                # KENDALA EXISTING:
                csp.addConstraint(lambda ns, ew: ns + ew + 2 * self.yellow_time <= 120, ('green_ns', 'green_ew'))
                
                # PEMERIKSAAN KENDALA, JIKA TIDAK ADA KONTRADIKSI MAKA AKAN LANJUT MEMILIH VARIABLE
                # JIKA TERJADI KONTRADIKSI, MAKA AKAN MUNDUR (BACKTRACK)
                if self.current_ns_waiting_time > 0 and self.current_ew_waiting_time > 0:
                    if self.current_ns_waiting_time > self.current_ew_waiting_time * 1.5:
                        csp.addConstraint(lambda ns, ew: ns >= ew * 1.05, ('green_ns', 'green_ew'))
                    elif self.current_ew_waiting_time > self.current_ns_waiting_time * 1.5:
                        csp.addConstraint(lambda ns, ew: ew >= ns * 1.05, ('green_ns', 'green_ew'))
                
                if self.current_ns_queue_length > 5 and self.current_ew_queue_length > 5:
                    csp.addConstraint(lambda ns, ew: abs(ns - ew) <= (self.max_green - self.min_green) / 2, ('green_ns', 'green_ew'))

                if self.current_ew_waiting_time >= 25: 
                    if self.current_ns_waiting_time > 5: 
                        csp.addConstraint(lambda ns, ew: ns >= max(ew * 0.3, self.min_green + 10), ('green_ns', 'green_ew'))
                
                if self.current_ns_waiting_time >= 25: 
                    if self.current_ew_queue_length > 0 or self.current_ew_waiting_time > 0: 
                        csp.addConstraint(lambda ns, ew: ew >= max(ns * 0.3, self.min_green + 10), ('green_ns', 'green_ew'))
                        csp.addConstraint(lambda ns, ew: ns <= self.max_green - 10, ('green_ns', 'green_ew'))

                # --- KENDALA BARU: Paksa NS ke lampu hijau minimum jika permintaannya sangat rendah ---
                if self.current_ns_waiting_time < 1.0 and self.current_ns_queue_length < 5: 
                    csp.addConstraint(lambda ns_val: ns_val == self.min_green, ('green_ns',))

                solution = csp.getSolution()
                
                if solution:
                    green_ns_final = solution['green_ns']
                    green_ew_final = solution['green_ew']
                else:
                    print(f"Warning: No CSP solution found at step {self.step}. Using fallback green times.")
                    green_ns_final = int(total_green_budget / 2) 
                    green_ew_final = int(total_green_budget / 2) 
                    green_ns_final = max(self.min_green, min(green_ns_final, self.max_green))
                    green_ew_final = max(self.min_green, min(green_ew_final, self.max_green))
                    
                print(f"Step {self.step}:")
                print(f"  NS Queue: {self.current_ns_queue_length}, NS Waiting Time: {self.current_ns_waiting_time:.2f}")
                print(f"  EW Queue: {self.current_ew_queue_length}, EW Waiting Time: {self.current_ew_waiting_time:.2f}")
                print(f"  Green NS Final: {green_ns_final}s, Green EW Final: {green_ew_final}s")
                if solution:
                    print(f"  CSP solution found.")
                else:
                    print(f"  No CSP solution found, using fallback values.")
                print("-" * 30) 

                self._run_phase(green_ns_final, 0) 
                self._run_phase(self.yellow_time, 1) 
                self._run_phase(green_ew_final, 2) 
                self._run_phase(self.yellow_time, 3) 

            if self.total_vehicles_departed > 0:
                avg_waiting_time = self.total_waiting_time / self.total_vehicles_departed
                total_travel_time = sum(self.vehicle_travel_times.values())
                avg_travel_time = total_travel_time / self.total_vehicles_departed if self.total_vehicles_departed > 0 else 0
                throughput = self.total_vehicles_departed / self.step if self.step > 0 else 0
                
                print(f"\n--- Simulation Summary ---")
                print(f"Simulation selesai pada {self.step}. Total kendaraan yang muncul : {self.total_vehicles_departed}")
                print(f"Total waiting time : {self.total_waiting_time:.2f}s, Rerata waktu tunggu per kendaraan : {avg_waiting_time:.2f}s")
                print(f"Total waktu travel : {total_travel_time:.2f}s, Rerata waktu travel per kendaraan : {avg_travel_time:.2f}s")
                print(f"Throughput : {throughput:.4f} vehicles/step")
            else:
                print("No vehicles departed during the simulation.")
        except Exception as e:
            print(f"Simulation terminated with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.env.close()
            print("TraCI connection closed successfully")
            sys.stdout.flush()

if __name__ == "__main__":
    csp = TrafficLightCSP()
    csp.run()