#include "compact_result_format.h"
#include "compact_result_test_utils.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    assert(ai::format_compact_points({}).empty());

    const std::vector<ai::CompactPointResult> cv{
        {0, 364, 28},
        {1, 302, 18},
    };
    assert(ai::format_compact_points(cv) == "0,364,28|1,302,18");

    const std::vector<ai::CompactPointResult> gaps_repeats_and_negative{
        {1, -231, 417},
        {1, -25, -75},
        {4, 0, 0},
    };
    const std::string text = ai::format_compact_points(gaps_repeats_and_negative);
    assert(text == "1,-231,417|1,-25,-75|4,0,0");
    assert(text.front() != '|' && text.back() != '|');
    assert(text.find_first_of("[]{} ") == std::string::npos);

    std::vector<compact_test::Point> parsed;
    assert(compact_test::parse(text, &parsed));
    assert(parsed.size() == 3);
    assert(parsed[0].id == 1 && parsed[0].x == -231 && parsed[0].y == 417);
    assert(parsed[1].id == 1);
    assert(parsed[2].id == 4);

    for (const char* invalid : {"|0,1,2", "0,1,2|", "0,1", "0,1,2,3", "x,1,2", "-1,1,2"}) {
        assert(!compact_test::parse(invalid, &parsed));
    }
    return 0;
}
