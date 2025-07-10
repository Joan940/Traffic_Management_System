import traci
import sys
from constraint import Problem, BacktrackingSolver
import numpy as np
from sumoenv import SumoEnv # Asumsikan SumoEnv tersedia dan dikonfigurasi dengan benar
import collections
import random

class TrafficLightCSP:
    def __init__(self):
        # Inisialisasi lingkungan SUMO
        self.env = SumoEnv(label='csp_sim', gui_f=True)
        self.tl_id = "gneJ00" # ID lampu lalu lintas
        # Jalur untuk arah Utara-Selatan dan Timur-Barat
        self.ns_lanes = ['-gneE0_0', '-gneE0_1', '-gneE0_2', '-gneE2_0', '-gneE2_1', '-gneE2_2']
        self.ew_lanes = ['-gneE1_0', '-gneE1_1', '-gneE1_2', '-gneE3_0', '-gneE3_1', '-gneE3_2']

        # Durasi fase lampu lalu lintas
        self.min_green = 20  # Waktu hijau minimum dalam detik
        self.max_green = 60  # Waktu hijau maksimum dalam detik
        self.yellow_time = 5 # Durasi fase kuning
        self.red_time = 0    # Fase merah (biasanya 0 karena kuning menangani transisi)

        # Metrik simulasi
        self.step = 0 # Langkah simulasi saat ini
        self.total_vehicles_departed = 0 # Total kendaraan yang telah menyelesaikan perjalanan
        self.total_waiting_time = 0.0 # Waktu tunggu akumulatif semua kendaraan
        self.vehicle_travel_times = {} # Menyimpan waktu tempuh untuk setiap kendaraan yang berangkat
        self.vehicle_departure_times = {} # Menyimpan waktu keberangkatan untuk setiap kendaraan

        # Metrik lalu lintas saat ini untuk logging dan keadaan RL
        self.current_ns_waiting_time = 0.0
        self.current_ew_waiting_time = 0.0
        self.current_ns_queue_length = 0
        self.current_ew_queue_length = 0

        # Langkah simulasi maksimum
        self.max_simulation_steps = 500

        # Reinforcement Learning (RL) Parameters
        # PASTIKAN BAGIAN INI ADA DI DALAM __init__
        self.learning_rate = 0.1  # Alpha: Seberapa banyak informasi baru menimpa informasi lama
        self.discount_factor = 0.9  # Gamma: Pentingnya hadiah di masa depan
        self.exploration_rate = 1.0  # Epsilon: Probabilitas memilih tindakan acak (eksplorasi)
        self.min_epsilon = 0.01     # Tingkat eksplorasi minimum
        self.epsilon_decay_rate = 0.995 # Tingkat penurunan epsilon per langkah

        # Tindakan RL: Penyesuaian waktu hijau target (penyesuaian NS, penyesuaian EW) dalam detik
        # Penyesuaian ini akan diterapkan pada waktu hijau berbasis permintaan sebelum CSP
        self.actions = [
            (0, 0),    # Tidak ada perubahan
            (5, 0),    # Tingkatkan hijau NS sebesar 5 detik
            (-5, 0),   # Kurangi hijau NS sebesar 5 detik
            (0, 5),    # Tingkatkan hijau EW sebesar 5 detik
            (0, -5),   # Kurangi hijau EW sebesar 5 detik
            (5, 5),    # Tingkatkan keduanya sebesar 5 detik
            (-5, -5),  # Kurangi keduanya sebesar 5 detik
            (5, -5),   # Tingkatkan NS, Kurangi EW sebesar 5 detik
            (-5, 5)    # Kurangi NS, Tingkatkan EW sebesar 5 detik
        ]
        self.num_actions = len(self.actions)
        # Q-table: Menyimpan nilai-Q untuk pasangan (keadaan, tindakan).
        # defaultdict memungkinkan keadaan baru diinisialisasi dengan nol untuk semua tindakan.
        self.q_table = collections.defaultdict(lambda: np.zeros(self.num_actions))

    def _update_vehicle_metrics(self):
        """
        Memperbarui metrik terkait keberangkatan kendaraan dan waktu tempuh.
        Dipanggil pada setiap langkah simulasi.
        """
        # Melacak waktu keberangkatan untuk kendaraan baru
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in self.vehicle_departure_times:
                self.vehicle_departure_times[veh_id] = self.step

        # Menghitung waktu tempuh untuk kendaraan yang telah tiba
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            if veh_id in self.vehicle_departure_times:
                travel_time = self.step - self.vehicle_departure_times[veh_id]
                self.vehicle_travel_times[veh_id] = travel_time
                del self.vehicle_departure_times[veh_id] # Hapus dari pelacakan setelah tiba

    def _get_current_lane_metrics(self):
        """
        Menghitung dan memperbarui panjang antrian dan waktu tunggu saat ini untuk jalur NS dan EW.
        Metrik ini digunakan untuk perhitungan permintaan CSP dan definisi keadaan RL.
        """
        ns_queue = 0
        ew_queue = 0
        ns_waiting_sum = 0.0
        ew_waiting_sum = 0.0
        ns_vehicle_count_for_avg_wait = 0
        ew_vehicle_count_for_avg_wait = 0

        # Menggabungkan metrik untuk jalur Utara-Selatan
        for lane in self.ns_lanes:
            ns_queue += traci.lane.getLastStepHaltingNumber(lane) # Jumlah kendaraan yang berhenti
            current_lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in current_lane_vehicles:
                ns_waiting_sum += traci.vehicle.getWaitingTime(veh_id)
                ns_vehicle_count_for_avg_wait += 1

        # Menggabungkan metrik untuk jalur Timur-Barat
        for lane in self.ew_lanes:
            ew_queue += traci.lane.getLastStepHaltingNumber(lane)
            current_lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in current_lane_vehicles:
                ew_waiting_sum += traci.vehicle.getWaitingTime(veh_id)
                ew_vehicle_count_for_avg_wait += 1

        # Menghitung waktu tunggu rata-rata
        self.current_ns_waiting_time = (ns_waiting_sum / ns_vehicle_count_for_avg_wait) if ns_vehicle_count_for_avg_wait > 0 else 0.0
        self.current_ew_waiting_time = (ew_waiting_sum / ew_vehicle_count_for_avg_wait) if ew_vehicle_count_for_avg_wait > 0 else 0.0

        # Memperbarui panjang antrian saat ini
        self.current_ns_queue_length = ns_queue
        self.current_ew_queue_length = ew_queue

    def _run_phase(self, phase_duration, phase_id):
        """
        Menjalankan satu fase lampu lalu lintas untuk durasi yang diberikan.
        Memajukan simulasi SUMO langkah demi langkah.
        """
        self.env.set_traffic_light_phase(phase_id, phase_duration)
        for _ in range(int(phase_duration)):
            # Berhenti jika langkah simulasi maksimum tercapai
            if self.step >= self.max_simulation_steps:
                break

            self.env.simulation_step() # Majukan simulasi SUMO satu langkah
            self._update_vehicle_metrics() # Perbarui metrik pelacakan kendaraan

            # Dapatkan total kendaraan yang berhenti dan waktu tunggu untuk logging
            total_halting_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self.ns_lanes + self.ew_lanes)
            current_total_waiting_time_step = self.env.get_waiting_time()
            self.total_waiting_time += current_total_waiting_time_step

            # Perbarui metrik jalur saat ini untuk logging real-time di dalam fase
            self._get_current_lane_metrics()

            # Tulis data langkah simulasi saat ini ke file log
            with open('queue_length.txt', 'a') as f:
                f.write(f"{self.step},{total_halting_vehicles},{current_total_waiting_time_step},"
                        f"{self.current_ns_queue_length},{self.current_ew_queue_length},"
                        f"{self.current_ns_waiting_time:.2f},{self.current_ew_waiting_time:.2f}\n")
            self.step += 1 # Tambah penghitung langkah simulasi

    # --- Metode Pembantu Reinforcement Learning (RL) ---

    def _discretize_value(self, value, bins):
        """
        Mendiskretisasi nilai kontinu ke dalam bin yang telah ditentukan.
        Digunakan untuk mengubah metrik lalu lintas kontinu menjadi keadaan diskrit untuk RL.
        """
        for i, upper_bound in enumerate(bins):
            if value <= upper_bound:
                return i
        return len(bins) # Mengembalikan indeks bin terakhir jika nilai melebihi semua batas atas

    def _get_state(self):
        """
        Mengembalikan keadaan diskritisasi saat ini untuk agen RL.
        Keadaan adalah tuple yang merepresentasikan panjang antrian dan waktu tunggu yang telah dibin.
        """
        # Tentukan bin untuk panjang antrian dan waktu tunggu. Ini bisa disesuaikan.
        # Example: [5, 15, 30] means bins for (0-5], (5-15], (15-30], (30+)
        ns_q_bin = self._discretize_value(self.current_ns_queue_length, [5, 15, 30])
        ew_q_bin = self._discretize_value(self.current_ew_queue_length, [5, 15, 30])
        ns_w_bin = self._discretize_value(self.current_ns_waiting_time, [10, 30, 60])
        ew_w_bin = self._discretize_value(self.current_ew_waiting_time, [10, 30, 60])
        return (ns_q_bin, ew_q_bin, ns_w_bin, ew_w_bin)

    def _choose_action(self, state):
        """
        Memilih tindakan menggunakan strategi epsilon-greedy.
        Dengan probabilitas epsilon, tindakan acak dipilih (eksplorasi).
        Jika tidak, tindakan dengan nilai-Q tertinggi untuk keadaan saat ini dipilih (eksploitasi).
        """
        if random.uniform(0, 1) < self.exploration_rate: # Menggunakan self.exploration_rate (epsilon)
            return random.randrange(self.num_actions) # Eksplorasi: pilih indeks tindakan acak
        else:
            return np.argmax(self.q_table[state]) # Eksploitasi: pilih tindakan dengan nilai-Q tertinggi

    def _calculate_reward(self):
        """
        Menghitung hadiah berdasarkan metrik lalu lintas saat ini.
        Fungsi hadiah bertujuan untuk meminimalkan waktu tunggu dan panjang antrian.
        Nilai negatif digunakan karena Q-learning memaksimalkan hadiah, dan kita ingin meminimalkan metrik ini.
        Bobot (misalnya, 1.0 untuk waktu tunggu, 0.5 untuk panjang antrian) dapat disesuaikan untuk memprioritaskan metrik tertentu.
        """
        reward = - (self.current_ns_waiting_time * 1.0 + self.current_ew_waiting_time * 1.0) - \
                 (self.current_ns_queue_length * 0.5 + self.current_ew_queue_length * 0.5)
        return reward

    def _update_q_table(self, state, action_index, reward, next_state):
        """
        Memperbarui tabel-Q menggunakan formula Q-learning.
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
        """
        old_value = self.q_table[state][action_index]
        next_max = np.max(self.q_table[next_state]) # Nilai-Q maksimum untuk keadaan berikutnya
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action_index] = new_value

    def run(self):
        """
        Loop simulasi utama.
        Mengintegrasikan pengambilan keputusan RL dengan pemenuhan batasan CSP.
        """
        self.env.reset() # Atur ulang lingkungan simulasi SUMO

        # Inisialisasi/bersihkan file log
        with open('queue_length.txt', 'w') as f:
            f.write("step,total_halting_vehicles,total_waiting_time_step,ns_queue,ew_queue,ns_avg_wait_current,ew_avg_wait_current\n")

        try:
            while self.step < self.max_simulation_steps:
                # 1. Dapatkan metrik lalu lintas saat ini untuk keadaan saat ini (sebelum keputusan RL/CSP)
                self._get_current_lane_metrics()
                current_state = self._get_state()

                # 2. Agen RL memilih tindakan (penyesuaian waktu hijau) berdasarkan keadaan saat ini
                action_index = self._choose_action(current_state)
                adjustment_ns, adjustment_ew = self.actions[action_index]

                # 3. Hitung waktu hijau target CSP awal berdasarkan rasio permintaan saat ini
                ns_demand_metric = self.current_ns_queue_length + (self.current_ns_waiting_time * 5)
                ew_demand_metric = self.current_ew_queue_length + (self.current_ew_waiting_time * 5)

                # Pastikan metrik permintaan setidaknya 1 untuk menghindari pembagian dengan nol
                ns_demand_metric = max(ns_demand_metric, 1)
                ew_demand_metric = max(ew_demand_metric, 1)

                total_demand = ns_demand_metric + ew_demand_metric
                ns_ratio = ns_demand_metric / total_demand
                ew_ratio = ew_demand_metric / total_demand

                total_green_budget = 120 - (2 * self.yellow_time) # Total waktu hijau yang tersedia dalam satu siklus
                target_green_ns = int(total_green_budget * ns_ratio)
                target_green_ew = int(total_green_budget * ew_ratio)

                # 4. Terapkan penyesuaian RL ke waktu hijau target berbasis permintaan
                # Pastikan target yang disesuaikan tetap dalam batas waktu hijau min/maks
                adjusted_target_green_ns = max(self.min_green, min(self.max_green, target_green_ns + adjustment_ns))
                adjusted_target_green_ew = max(self.min_green, min(self.max_green, target_green_ew + adjustment_ew))

                # 5. CSP menyelesaikan waktu hijau akhir, dipandu oleh target yang disesuaikan RL
                csp = Problem(BacktrackingSolver(forwardcheck=True))

                # Tentukan domain variabel untuk green_ns dan green_ew.
                # Domain dipusatkan di sekitar target yang disesuaikan RL, dengan buffer +/- 10 detik.
                csp.addVariable('green_ns', range(max(self.min_green, adjusted_target_green_ns - 10), min(self.max_green, adjusted_target_green_ns + 10) + 1))
                csp.addVariable('green_ew', range(max(self.min_green, adjusted_target_green_ew - 10), min(self.max_green, adjusted_target_green_ew + 10) + 1))

                # Batasan CSP yang ada:
                # Pastikan total panjang siklus tidak melebihi 120 detik
                csp.addConstraint(lambda ns, ew: ns + ew + 2 * self.yellow_time <= 120, ('green_ns', 'green_ew'))
                # Pastikan kedua fase mendapatkan setidaknya waktu hijau minimum gabungan mereka
                csp.addConstraint(lambda ns, ew: ns + ew >= 2 * self.min_green, ('green_ns', 'green_ew'))

                # Batasan berdasarkan waktu tunggu saat ini untuk menyeimbangkan lalu lintas
                if self.current_ns_waiting_time > 0 and self.current_ew_waiting_time > 0:
                    if self.current_ns_waiting_time > self.current_ew_waiting_time * 1.5:
                        # Jika waktu tunggu NS jauh lebih tinggi, prioritaskan hijau NS
                        csp.addConstraint(lambda ns, ew: ns >= ew * 1.05, ('green_ns', 'green_ew'))
                    elif self.current_ew_waiting_time > self.current_ns_waiting_time * 1.5:
                        # Jika waktu tunggu EW jauh lebih tinggi, prioritaskan hijau EW
                        csp.addConstraint(lambda ns, ew: ew >= ns * 1.05, ('green_ns', 'green_ew'))

                # Batasan berdasarkan panjang antrian untuk memastikan keadilan
                if self.current_ns_queue_length > 5 and self.current_ew_queue_length > 5:
                    # Jaga agar waktu hijau relatif seimbang jika kedua antrian signifikan
                    csp.addConstraint(lambda ns, ew: abs(ns - ew) <= (self.max_green - self.min_green) / 2, ('green_ns', 'green_ew'))

                # Batasan khusus untuk waktu tunggu yang sangat tinggi
                if self.current_ew_waiting_time >= 25:
                    if self.current_ns_waiting_time > 5:
                        # Jika waktu tunggu EW sangat tinggi, pastikan NS mendapatkan minimum yang wajar
                        csp.addConstraint(lambda ns, ew: ns >= max(ew * 0.3, self.min_green + 10), ('green_ns', 'green_ew'))

                if self.current_ns_waiting_time >= 25:
                    if self.current_ew_queue_length > 0 or self.current_ew_waiting_time > 0:
                        # Jika waktu tunggu NS sangat tinggi, pastikan EW mendapatkan minimum yang wajar dan NS tidak terlalu lama
                        csp.addConstraint(lambda ns, ew: ew >= max(ns * 0.3, self.min_green + 10), ('green_ns', 'green_ew'))
                        csp.addConstraint(lambda ns, ew: ns <= self.max_green - 10, ('green_ns', 'green_ew'))

                # Batasan baru: Paksa NS ke lampu hijau minimum jika permintaan sangat rendah
                if self.current_ns_waiting_time < 1.0 and self.current_ns_queue_length < 5:
                    csp.addConstraint(lambda ns_val: ns_val == self.min_green, ('green_ns',))
                # Batasan baru: Paksa EW ke lampu hijau minimum jika permintaan sangat rendah
                if self.current_ew_waiting_time < 1.0 and self.current_ew_queue_length < 5:
                    csp.addConstraint(lambda ew_val: ew_val == self.min_green, ('green_ew',))

                # Coba temukan solusi untuk CSP
                solution = csp.getSolution()

                if solution:
                    green_ns_final = solution['green_ns']
                    green_ew_final = solution['green_ew']
                else:
                    # Cadangan jika tidak ada solusi CSP yang ditemukan (seharusnya jarang dengan batasan yang terdefinisi dengan baik)
                    print(f"Peringatan: Tidak ada solusi CSP ditemukan pada langkah {self.step}. Menggunakan cadangan ke target yang disesuaikan.")
                    green_ns_final = adjusted_target_green_ns
                    green_ew_final = adjusted_target_green_ew

                # Cetak keputusan dan metrik siklus saat ini
                print(f"Langkah {self.step}:")
                print(f"  Antrian NS: {self.current_ns_queue_length}, Waktu Tunggu NS: {self.current_ns_waiting_time:.2f}")
                print(f"  Antrian EW: {self.current_ew_queue_length}, Waktu Tunggu EW: {self.current_ew_waiting_time:.2f}")
                print(f"  Penyesuaian RL: NS={adjustment_ns}s, EW={adjustment_ew}s (Indeks Tindakan: {action_index})")
                print(f"  Target NS (berbasis Permintaan): {target_green_ns}s, Target EW (berbasis Permintaan): {target_green_ew}s")
                print(f"  Target NS yang disesuaikan: {adjusted_target_green_ns}s, Target EW yang disesuaikan: {adjusted_target_green_ew}s")
                print(f"  Hijau NS Akhir: {green_ns_final}s, Hijau EW Akhir: {green_ew_final}s")
                if solution:
                    print(f"  Solusi CSP ditemukan.")
                else:
                    print(f"  Tidak ada solusi CSP ditemukan, menggunakan nilai cadangan.")
                print(f"  Epsilon (Tingkat Eksplorasi): {self.exploration_rate:.4f}") # Menggunakan self.exploration_rate
                print("-" * 30)

                # 6. Jalankan fase lampu lalu lintas dengan waktu hijau yang ditentukan
                self._run_phase(green_ns_final, 0) # Fase Hijau NS
                if self.step >= self.max_simulation_steps: break # Periksa apakah simulasi berakhir selama fase
                self._run_phase(self.yellow_time, 1) # Fase Kuning NS
                if self.step >= self.max_simulation_steps: break
                self._run_phase(green_ew_final, 2) # Fase Hijau EW
                if self.step >= self.max_simulation_steps: break
                self._run_phase(self.yellow_time, 3) # Fase Kuning EW
                if self.step >= self.max_simulation_steps: break

                # 7. Dapatkan keadaan berikutnya dan hitung hadiah untuk pembaruan RL
                self._get_current_lane_metrics() # Perbarui metrik setelah siklus penuh untuk next_state
                next_state = self._get_state()
                reward = self._calculate_reward()

                # 8. Perbarui tabel-Q menggunakan pengalaman yang diamati
                self._update_q_table(current_state, action_index, reward, next_state)

                # 9. Kurangi epsilon untuk secara bertahap mengurangi eksplorasi
                self.exploration_rate = max(self.min_epsilon, self.exploration_rate * self.epsilon_decay_rate)

            # --- Ringkasan Simulasi setelah loop utama selesai ---
            # Hitung ulang total kendaraan yang berangkat, karena beberapa mungkin telah tiba di siklus terakhir
            self.total_vehicles_departed = len(self.vehicle_travel_times)
            if self.total_vehicles_departed > 0:
                avg_waiting_time = self.total_waiting_time / self.total_vehicles_departed
                total_travel_time = sum(self.vehicle_travel_times.values())
                avg_travel_time = total_travel_time / self.total_vehicles_departed if self.total_vehicles_departed > 0 else 0
                throughput = self.total_vehicles_departed / self.step if self.step > 0 else 0

                print(f"\n--- Ringkasan Simulasi (Lampu Lalu Lintas Adaptif CSP + RL) ---")
                print(f"Simulasi berakhir pada langkah {self.step}. Total kendaraan berangkat: {self.total_vehicles_departed}")
                print(f"Total waktu tunggu: {self.total_waiting_time:.2f}s, Waktu tunggu rata-rata per kendaraan: {avg_waiting_time:.2f}s")
                print(f"Total waktu tempuh: {total_travel_time:.2f}s, Waktu tempuh rata-rata per kendaraan: {avg_travel_time:.2f}s")
                print(f"Throughput: {throughput:.4f} kendaraan/langkah")
            else:
                print("Tidak ada kendaraan yang berangkat selama simulasi.")
        except Exception as e:
            print(f"Simulasi dihentikan dengan kesalahan: {e}")
            import traceback
            traceback.print_exc() # Cetak traceback lengkap untuk debugging
        finally:
            self.env.close() # Tutup koneksi lingkungan SUMO
            print("Koneksi TraCI berhasil ditutup")
            sys.stdout.flush() # Pastikan semua pernyataan print sudah di-flush ke konsol

if __name__ == "__main__":
    csp = TrafficLightCSP()
    csp.run()