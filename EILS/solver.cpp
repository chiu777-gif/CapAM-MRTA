// Paper algorithms, continuous-time TAPTC adaptation. See SOURCES.md.
// Build without fast-math: deadline boundary comparisons must remain exact.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {
struct Candidate {
    double shift = std::numeric_limits<double>::infinity();
    int position = -1;
};

struct Search {
    int n, m;
    const double *deadline;
    std::vector<double> travel, start_travel, service;
    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> finish, slack;
    std::vector<std::vector<Candidate>> cache;
    std::vector<bool> used;

    Search(int tasks, int robots, const double *xy, const double *starts,
           const double *due, const double *work, const double *capacity,
           double speed, bool discrete)
        : n(tasks), m(robots), deadline(due), travel(n * n),
          start_travel(m * n), service(m * n), tours(m), finish(m), slack(m),
          cache(m, std::vector<Candidate>(n)), used(n, false) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                travel[i*n+j] = std::hypot(xy[2*i]-xy[2*j], xy[2*i+1]-xy[2*j+1]) / speed;
            for (int a = 0; a < m; ++a) {
                start_travel[a*n+i] = std::hypot(starts[2*a]-xy[2*i], starts[2*a+1]-xy[2*i+1]) / speed;
                double value = work[i] / capacity[a];
                service[a*n+i] = discrete ? std::ceil(value) : value;
            }
        }
    }

    double leg(int a, int previous, int task) const {
        return previous < 0 ? start_travel[a*n+task] : travel[previous*n+task];
    }

    void times(int a) {
        const auto &tour = tours[a];
        int size = static_cast<int>(tour.size());
        finish[a].resize(size);
        slack[a].resize(size);
        double time = 0;
        int previous = -1;
        for (int p = 0; p < size; ++p) {
            int task = tour[p];
            time += leg(a, previous, task) + service[a*n+task];
            finish[a][p] = time;
            previous = task;
        }
        double minimum = std::numeric_limits<double>::infinity();
        for (int p = size - 1; p >= 0; --p) {
            minimum = std::min(minimum, deadline[tour[p]] - finish[a][p]);
            slack[a][p] = minimum;  // Eq. (7), all visits in the tour are feasible.
        }
    }

    void candidates(int a) {
        const auto &tour = tours[a];
        int size = static_cast<int>(tour.size());
        for (int task = 0; task < n; ++task) {
            cache[a][task] = Candidate{};
            if (used[task]) continue;
            Candidate best;
            for (int p = 0; p <= size; ++p) {
                int previous = p == 0 ? -1 : tour[p-1];
                double increment = leg(a, previous, task) + service[a*n+task];
                double completion = (p == 0 ? 0.0 : finish[a][p-1]) + increment;
                if (completion > deadline[task]) continue;
                double shift = increment;
                if (p < size) {
                    int next = tour[p];
                    shift += travel[task*n+next] - leg(a, previous, next);  // Eq. (6).
                    if (shift > slack[a][p]) continue;  // Eq. (8).
                }
                if (shift < best.shift) best = Candidate{shift, p};
            }
            cache[a][task] = best;
        }
    }

    int construct() {
        int score = 0;
        for (int a = 0; a < m; ++a) {
            times(a);
            candidates(a);
            score += static_cast<int>(tours[a].size());
        }
        while (score < n) {
            int chosen = -1, robot = -1, position = -1;
            double minimum = std::numeric_limits<double>::infinity();
            // Equal insertion shifts: ascending task, robot, position.
            for (int task = 0; task < n; ++task) {
                if (used[task]) continue;
                for (int a = 0; a < m; ++a) {
                    const auto &candidate = cache[a][task];
                    if (candidate.shift < minimum) {
                        minimum = candidate.shift;
                        chosen = task;
                        robot = a;
                        position = candidate.position;
                    }
                }
            }
            if (chosen < 0) break;
            tours[robot].insert(tours[robot].begin() + position, chosen);
            used[chosen] = true;
            ++score;
            times(robot);
            candidates(robot);
        }
        return score;
    }

    void shake(int number, int start, bool random_start, std::mt19937_64 &rng) {
        for (int a = 0; a < m; ++a) {
            auto &tour = tours[a];
            int size = static_cast<int>(tour.size());
            if (size == 0) continue;
            int x = random_start ? static_cast<int>(rng() % size) : start % size;
            int count = std::min(number, size);
            std::vector<int> kept;
            for (int p = 0; p < size; ++p) {
                if ((p - x + size) % size < count) used[tour[p]] = false;
                else kept.push_back(tour[p]);
            }
            tour.swap(kept);
        }
    }

    void complete_all(double *output_finish, int *output_robot, int *output_order,
                      int *output_lengths) {
        std::fill(used.begin(), used.end(), false);
        std::vector<double> ready(m, 0.0);
        std::vector<int> previous(m, -1);
        for (int a = 0; a < m; ++a) {
            for (int task : tours[a]) {
                ready[a] += leg(a, previous[a], task) + service[a*n+task];
                output_finish[task] = ready[a];
                output_robot[task] = a;
                previous[a] = task;
                used[task] = true;
            }
        }
        // Shared explicit CapAM cost extension: reserve the feasible tours,
        // then append remaining tasks to the earliest-free robot, choosing
        // minimum travel+service time. This does not change the search score.
        int assigned = 0;
        for (bool value : used) assigned += value;
        while (assigned < n) {
            int a = static_cast<int>(std::min_element(ready.begin(), ready.end()) - ready.begin());
            int chosen = -1;
            double minimum = std::numeric_limits<double>::infinity();
            for (int task = 0; task < n; ++task) {
                if (used[task]) continue;
                double increment = leg(a, previous[a], task) + service[a*n+task];
                if (increment < minimum) { chosen = task; minimum = increment; }
            }
            ready[a] += minimum;
            output_finish[chosen] = ready[a];
            output_robot[chosen] = a;
            previous[a] = chosen;
            tours[a].push_back(chosen);
            used[chosen] = true;
            ++assigned;
        }
        int cursor = 0;
        for (int a = 0; a < m; ++a) {
            output_lengths[a] = static_cast<int>(tours[a].size());
            for (int task : tours[a]) output_order[cursor++] = task;
        }
    }
};
}  // namespace

// algorithm: 0 = construction only; 1 = ILS; 2 = enhanced ILS.
extern "C" int solve_taptc(
    int n, int m, const double *xy, const double *starts, const double *deadline,
    const double *work, const double *capacity, double speed, int algorithm,
    int limit, double coefficient, std::uint64_t seed, int discrete,
    double *out_finish, int *out_robot, int *out_order, int *out_lengths,
    int *out_stats) {
    if (n < 1 || m < 1 || algorithm < 0 || algorithm > 2 || limit < 1 ||
        !(coefficient > 0 && coefficient <= 1) || !(speed > 0)) return 1;
    try {
        Search search(n, m, xy, starts, deadline, work, capacity, speed, discrete != 0);
        std::mt19937_64 rng(seed);
        std::vector<std::vector<int>> best(m);
        int best_score = -1, iterations = 0, best_iteration = 0, trials = 0;
        int remove = 1, start = 0;
        // ILS' threshold depends on total tasks, not current score.
        int threshold = std::max(1, static_cast<int>(std::floor(n * coefficient / m)));
        while (algorithm == 2 ? iterations < limit : (algorithm == 0 ? iterations < 1 : trials < limit)) {
            ++iterations;
            int score = search.construct();
            if (score > best_score) {
                best_score = score;
                best = search.tours;
                best_iteration = iterations;
                trials = 0;
                remove = 1;
            } else ++trials;
            if (score == n || algorithm == 0) break;
            if (algorithm == 2) {
                // Algorithm 2 does not specify integer rounding or r=0.
                int count = std::max(1, static_cast<int>(std::floor(score * coefficient / m)));
                search.shake(count, 0, true, rng);
            } else {
                search.shake(remove, start, false, rng);
                start += remove;
                ++remove;
                int shortest = n;
                for (const auto &tour : search.tours)
                    shortest = std::min(shortest, static_cast<int>(tour.size()));
                start = shortest > 0 ? start % shortest : 0;
                if (remove >= threshold) remove = 1;
            }
        }
        search.tours = best;
        search.complete_all(out_finish, out_robot, out_order, out_lengths);
        out_stats[0] = best_score;
        out_stats[1] = iterations;
        out_stats[2] = best_iteration;
        return 0;
    } catch (...) { return 2; }
}
