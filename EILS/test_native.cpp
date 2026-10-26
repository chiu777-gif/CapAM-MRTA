// Exercise circular removal directly, including wraparound and empty tours.
// Compile this translation unit alone; solver.cpp is intentionally included.
#include "solver.cpp"
#include <cassert>

int main() {
    const double xy[] = {0,0, 1,0, 2,0, 3,0, 4,0, 5,0};
    const double starts[] = {0,0, 0,0};
    const double deadlines[] = {20,20,20,20,20,20};
    const double workload[] = {1,1,1,1,1,1};
    const double capacities[] = {1,1};
    Search search(6, 2, xy, starts, deadlines, workload, capacities, 1, false);
    search.tours[0] = {0,1,2,3,4,5};
    std::fill(search.used.begin(), search.used.end(), true);
    std::mt19937_64 rng(123);
    search.shake(3, 4, false, rng);
    assert((search.tours[0] == std::vector<int>{1,2,3}));
    assert(search.tours[1].empty());
    assert(!search.used[0] && !search.used[4] && !search.used[5]);
    assert(search.used[1] && search.used[2] && search.used[3]);
    search.times(0);
    assert((search.finish[0] == std::vector<double>{2,4,6}));
    assert((search.slack[0] == std::vector<double>{14,14,14}));
    search.shake(99, 99, false, rng);
    assert(search.tours[0].empty());
    assert(std::none_of(search.used.begin(), search.used.end(), [](bool b) { return b; }));
    // Removing an interior segment preserves original order and shifts visits earlier.
    search.tours[0] = {0,1,2,3,4,5};
    std::fill(search.used.begin(), search.used.end(), true);
    search.shake(2, 2, false, rng);
    assert((search.tours[0] == std::vector<int>{0,1,4,5}));
    search.times(0);
    assert((search.finish[0] == std::vector<double>{1,3,7,9}));
    // Random starts remain within each route and remove exactly r unique visits.
    for (int trial = 0; trial < 100; ++trial) {
        search.tours[0] = {0,1,2,3,4,5};
        std::fill(search.used.begin(), search.used.end(), true);
        search.shake(2, 0, true, rng);
        assert(search.tours[0].size() == 4);
        assert(std::count(search.used.begin(), search.used.end(), true) == 4);
    }
}
