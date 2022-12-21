#include <cstdio>
#include <cstdint>
#include <cassert>
#include <filesystem>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;

const unsigned NUM_ROW = 3;
const unsigned NUM_COL = NUM_ROW;
const unsigned NUM_PLATE = NUM_ROW * NUM_COL;
const unsigned STOP_CONDITION = 5;

const string STATE_SAVE_FILE_PATH = "../dat/ff14_minipot.csv";

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

struct MiniPotSolver
{
    unordered_map<unsigned, unsigned> rotate_90d_plate_idx_lookup;
    unordered_map<unsigned, unsigned> flip_horizontal_plate_idx_lookup;
    unordered_map<unsigned, unsigned> flip_vertical_plate_idx_lookup;

    unordered_map<unsigned, unsigned> rotate_90d_line_idx_lookup;
    unordered_map<unsigned, unsigned> flip_horizontal_line_idx_lookup;
    unordered_map<unsigned, unsigned> flip_vertical_line_idx_lookup;

    unordered_map<string, double> dp_concealed_states;
    // `dp_disclosed_states` can better help us know what happened in the DP process.
    // unordered_map<pair<string, unsigned>, double, pair_hash> dp_disclosed_states;
    unordered_map<string, vector<unsigned>> dp_choice_states;

    void generate_rotate_90d_lookups()
    {
        for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
        {
            for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
            {
                unsigned plate_idx = row_idx * NUM_COL + col_idx;
                unsigned n_plate_idx = col_idx * NUM_COL + (NUM_COL - 1 - row_idx);
                rotate_90d_plate_idx_lookup[plate_idx] = n_plate_idx;
            }
        }
        for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
        {
            rotate_90d_line_idx_lookup[row_idx] = NUM_ROW + (NUM_COL - 1 - row_idx);
        }
        for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
        {
            rotate_90d_line_idx_lookup[NUM_ROW + col_idx] = col_idx;
        }
        rotate_90d_line_idx_lookup[NUM_ROW + NUM_COL] = NUM_ROW + NUM_COL + 1;
        rotate_90d_line_idx_lookup[NUM_ROW + NUM_COL + 1] = NUM_ROW + NUM_COL;
    }

    void generate_flip_horizontal_lookups()
    {
        for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
        {
            for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
            {
                unsigned plate_idx = row_idx * NUM_COL + col_idx;
                unsigned n_plate_idx = row_idx * NUM_COL + (NUM_COL - 1 - col_idx);
                flip_horizontal_plate_idx_lookup[plate_idx] = n_plate_idx;
            }
        }
        for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
        {
            flip_horizontal_line_idx_lookup[row_idx] = row_idx;
        }
        for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
        {
            flip_horizontal_line_idx_lookup[NUM_ROW + col_idx] = NUM_ROW + (NUM_COL - 1 - col_idx);
        }
        flip_horizontal_line_idx_lookup[NUM_ROW + NUM_COL] = NUM_ROW + NUM_COL + 1;
        flip_horizontal_line_idx_lookup[NUM_ROW + NUM_COL + 1] = NUM_ROW + NUM_COL;
    }

    void generate_flip_vertical_lookups()
    {
        for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
        {
            for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
            {
                unsigned plate_idx = row_idx * NUM_COL + col_idx;
                unsigned n_plate_idx = (NUM_ROW - 1 - row_idx) * NUM_COL + col_idx;
                flip_vertical_plate_idx_lookup[plate_idx] = n_plate_idx;
            }
        }
        for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
        {
            flip_vertical_line_idx_lookup[row_idx] = NUM_ROW - 1 - row_idx;
        }
        for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
        {
            flip_vertical_line_idx_lookup[NUM_ROW + col_idx] = NUM_ROW + col_idx;
        }
        flip_vertical_line_idx_lookup[NUM_ROW + NUM_COL] = NUM_ROW + NUM_COL + 1;
        flip_vertical_line_idx_lookup[NUM_ROW + NUM_COL + 1] = NUM_ROW + NUM_COL;
    }

    MiniPotSolver()
    {
        generate_rotate_90d_lookups();
        generate_flip_horizontal_lookups();
        generate_flip_vertical_lookups();
    }

    double get_reward(unsigned int score)
    {
        switch (score)
        {
        case 6:
            return 10000;
        case 7:
            return 36;
        case 8:
            return 720;
        case 9:
            return 360;
        case 10:
            return 80;
        case 11:
            return 252;
        case 12:
            return 108;
        case 13:
            return 72;
        case 14:
            return 54;
        case 15:
            return 180;
        case 16:
            return 72;
        case 17:
            return 180;
        case 18:
            return 119;
        case 19:
            return 36;
        case 20:
            return 306;
        case 21:
            return 1080;
        case 22:
            return 144;
        case 23:
            return 1800;
        case 24:
            return 3600;
        default:
            assert(false);
        }
    }

    void generate_complete_states(unsigned plate_idx, string state, vector<bool> &b_used_plate_idxes, vector<string> &completed_states)
    {
        if (plate_idx == NUM_PLATE)
        {
            completed_states.emplace_back(state);
            return;
        }

        if (state[plate_idx] != '0')
        {
            return generate_complete_states(plate_idx + 1, state, b_used_plate_idxes, completed_states);
        }

        char cached_c_plate_val = state[plate_idx];
        for (unsigned plate_val = 0; plate_val < NUM_PLATE; plate_val += 1)
        {
            if (!b_used_plate_idxes[plate_val])
            {
                char c_plate_val = '0' + plate_val + 1;
                state[plate_idx] = c_plate_val;
                b_used_plate_idxes[plate_val] = true;
                generate_complete_states(plate_idx + 1, state, b_used_plate_idxes, completed_states);
                b_used_plate_idxes[plate_val] = false;
            }
        }
        state[plate_idx] = cached_c_plate_val;
    }

    string transform(const string &state, const unordered_map<unsigned, unsigned> &plate_idx_lookup)
    {
        string n_state = state;
        for (auto key_val_pair : plate_idx_lookup)
        {
            n_state[key_val_pair.second] = state[key_val_pair.first];
        }
        return n_state;
    }

    vector<unsigned> transform(const vector<unsigned> choice_state, const unordered_map<unsigned, unsigned> &plate_idx_lookup)
    {
        vector<unsigned> n_choice_state = choice_state;
        for (unsigned choice_idx = 0; choice_idx < choice_state.size(); choice_idx += 1)
        {
            n_choice_state[choice_idx] = plate_idx_lookup.at(choice_state[choice_idx]);
        }
        sort(n_choice_state.begin(), n_choice_state.end());
        return n_choice_state;
    }

    double dp(string state, unsigned available_plate_cnt, vector<bool> &b_used_plate_idxes)
    {
        if (dp_concealed_states.find(state) != dp_concealed_states.end())
        {
            return dp_concealed_states[state];
        }

        double max_reward = 0;
        if (available_plate_cnt == STOP_CONDITION)
        {
            vector<string> completed_states;
            generate_complete_states(0, state, b_used_plate_idxes, completed_states);

            // row-wise; encoded from 0~2
            for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
            {
                double avg_reward = 0;
                for (auto completed_state : completed_states)
                {
                    unsigned score = 0;
                    for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
                    {
                        unsigned plate_idx = row_idx * NUM_COL + col_idx;
                        score += completed_state[plate_idx] - '0';
                    }
                    avg_reward += get_reward(score);
                }
                avg_reward /= completed_states.size();
                if (avg_reward > max_reward)
                {
                    max_reward = avg_reward;
                    dp_choice_states[state] = {row_idx};
                }
                else if (avg_reward == max_reward)
                {
                    dp_choice_states[state].push_back(row_idx);
                }
            }

            // col-wise; encoded from 3~5
            for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
            {
                double avg_reward = 0;
                for (auto completed_state : completed_states)
                {
                    unsigned score = 0;
                    for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
                    {
                        unsigned plate_idx = NUM_COL * row_idx + col_idx;
                        score += completed_state[plate_idx] - '0';
                    }
                    avg_reward += get_reward(score);
                }
                avg_reward /= completed_states.size();
                if (avg_reward > max_reward)
                {
                    max_reward = avg_reward;
                    dp_choice_states[state] = {NUM_ROW + col_idx};
                }
                else if (avg_reward == max_reward)
                {
                    dp_choice_states[state].push_back(NUM_ROW + col_idx);
                }
            }

            // main-diagnoal; encoded as 6
            {
                double avg_reward = 0;
                for (auto completed_state : completed_states)
                {
                    unsigned score = 0;
                    for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
                    {
                        unsigned plate_idx = NUM_COL * row_idx + row_idx;
                        score += completed_state[plate_idx] - '0';
                    }
                    avg_reward += get_reward(score);
                }
                avg_reward /= completed_states.size();
                if (avg_reward > max_reward)
                {
                    max_reward = avg_reward;
                    dp_choice_states[state] = {NUM_ROW + NUM_COL};
                }
                else if (avg_reward == max_reward)
                {
                    dp_choice_states[state].push_back(NUM_ROW + NUM_COL);
                }
            }

            // sub-diagnoal; encoded as 7
            {
                double avg_reward = 0;
                for (auto completed_state : completed_states)
                {
                    unsigned score = 0;
                    for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
                    {
                        unsigned plate_idx = NUM_COL * row_idx + (NUM_COL - 1 - row_idx);
                        score += completed_state[plate_idx] - '0';
                    }
                    avg_reward += get_reward(score);
                }
                avg_reward /= completed_states.size();
                if (avg_reward > max_reward)
                {
                    max_reward = avg_reward;
                    dp_choice_states[state] = {NUM_ROW + NUM_COL + 1};
                }
                else if (avg_reward == max_reward)
                {
                    dp_choice_states[state].push_back(NUM_ROW + NUM_COL + 1);
                }
            }
        }
        else
        {
            for (unsigned plate_idx = 0; plate_idx < NUM_PLATE; plate_idx += 1)
            {
                if (state[plate_idx] == '0')
                {
                    double avg_reward = 0;
                    for (unsigned plate_val = 0; plate_val < NUM_PLATE; plate_val += 1)
                    {
                        if (!b_used_plate_idxes[plate_val])
                        {
                            char c_plate_val = '0' + plate_val + 1;
                            state[plate_idx] = c_plate_val;
                            b_used_plate_idxes[plate_val] = true;
                            avg_reward += dp(state, available_plate_cnt - 1, b_used_plate_idxes);
                            b_used_plate_idxes[plate_val] = false;
                        }
                    }
                    state[plate_idx] = '0';
                    avg_reward /= available_plate_cnt;

                    // dp_disclosed_states[make_pair(state, plate_idx)] = avg_reward;

                    if (avg_reward > max_reward)
                    {
                        max_reward = avg_reward;
                        dp_choice_states[state] = {plate_idx};
                    }
                    else if (avg_reward == max_reward)
                    {
                        dp_choice_states[state].push_back(plate_idx);
                    }
                }
            }
        }

        // Update all three rotated states, to save repeated visits
        {
            string n_state = state;
            vector<unsigned> n_choice_state = dp_choice_states[state];
            for (unsigned rotate_idx = 1; rotate_idx < 4; rotate_idx += 1)
            {
                n_state = transform(n_state, rotate_90d_plate_idx_lookup);
                n_choice_state = available_plate_cnt == STOP_CONDITION
                                     ? transform(n_choice_state, rotate_90d_line_idx_lookup)
                                     : transform(n_choice_state, rotate_90d_plate_idx_lookup);
                dp_concealed_states[n_state] = max_reward;
                dp_choice_states[n_state] = n_choice_state;
            }
        }

        // Update flip horizontal state, to save repeated visits
        {
            string n_state = transform(state, flip_horizontal_plate_idx_lookup);
            vector<unsigned> n_choice_state = available_plate_cnt == STOP_CONDITION
                                                  ? transform(dp_choice_states[state], flip_horizontal_line_idx_lookup)
                                                  : transform(dp_choice_states[state], flip_horizontal_plate_idx_lookup);
            dp_concealed_states[n_state] = max_reward;
            dp_choice_states[n_state] = n_choice_state;
        }

        // Update flip veritical state, to save repeated visits
        {
            string n_state = transform(state, flip_vertical_plate_idx_lookup);
            vector<unsigned> n_choice_state = available_plate_cnt == STOP_CONDITION
                                                  ? transform(dp_choice_states[state], flip_vertical_line_idx_lookup)
                                                  : transform(dp_choice_states[state], flip_vertical_plate_idx_lookup);
            dp_concealed_states[n_state] = max_reward;
            dp_choice_states[n_state] = n_choice_state;
        }

        return dp_concealed_states[state] = max_reward;
    }

    void solve()
    {
        string state = string(NUM_PLATE, '0');
        vector<bool> b_used_plate_idxes(NUM_PLATE, false);
        dp(state, NUM_PLATE, b_used_plate_idxes);
    }

    void save_as_csv()
    {
        map<string, double> sorted_dp_concealed_states(dp_concealed_states.begin(), dp_concealed_states.end());

        FILE *pfile = fopen(STATE_SAVE_FILE_PATH.c_str(), "w");
        assert(pfile);
        fprintf(pfile, "state,avg_reward,best_choices\n");
        for (auto key_val_pair : sorted_dp_concealed_states)
        {
            string state = key_val_pair.first;
            double concealed_val = key_val_pair.second;
            fprintf(pfile, "\"%s\",%.10f,[", state.c_str(), concealed_val);
            for (unsigned choice_idx = 0; choice_idx < dp_choice_states[state].size(); choice_idx += 1)
            {
                fprintf(pfile, "%d%c", dp_choice_states[state][choice_idx], choice_idx == dp_choice_states[state].size() - 1 ? ']' : ',');
            }
            fprintf(pfile, "\n");
        }
        fclose(pfile);
    }
};

int main()
{
    auto mini_pot_solver = MiniPotSolver();
    mini_pot_solver.solve();
    mini_pot_solver.save_as_csv();
}