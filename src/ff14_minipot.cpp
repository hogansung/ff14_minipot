#include <cstdio>
#include <cstdint>
#include <cassert>
#include <filesystem>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>

using namespace std;

const unsigned NUM_ROW = 3;
const unsigned NUM_COL = 3;
const unsigned NUM_PLATE = NUM_ROW * NUM_COL;

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
    unordered_map<string, double> dp_concealed_states;
    unordered_map<pair<string, unsigned>, double, pair_hash> dp_disclosed_states;
    unordered_map<string, vector<unsigned>> dp_choice_states;

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

    MiniPotSolver() {}

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

    double dp(string state, unsigned available_plate_cnt, vector<bool> &b_used_plate_idxes)
    {
        if (dp_concealed_states.find(state) != dp_concealed_states.end())
        {
            return dp_concealed_states[state];
        }

        if (available_plate_cnt == 5)
        {
            vector<string> completed_states;
            generate_complete_states(0, state, b_used_plate_idxes, completed_states);

            double max_reward = 0;

            // row-wise; encoded from 0~2
            for (unsigned row_idx = 0; row_idx < NUM_ROW; row_idx += 1)
            {
                double avg_reward = 0;
                for (auto completed_state : completed_states)
                {
                    unsigned score = 0;
                    for (unsigned col_idx = 0; col_idx < NUM_COL; col_idx += 1)
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
                    dp_choice_states[state] = {6};
                }
                else if (avg_reward == max_reward)
                {
                    dp_choice_states[state].push_back(6);
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
                    dp_choice_states[state] = {7};
                }
                else if (avg_reward == max_reward)
                {
                    dp_choice_states[state].push_back(7);
                }
            }

            return dp_concealed_states[state] = max_reward;
        }

        double max_reward = 0;
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

                dp_disclosed_states[make_pair(state, plate_idx)] = avg_reward;

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
        FILE *pfile = fopen(STATE_SAVE_FILE_PATH.c_str(), "w");
        assert(pfile);
        fprintf(pfile, "state,avg_reward,best_choices\n");
        for (auto key_val_pair : dp_concealed_states)
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