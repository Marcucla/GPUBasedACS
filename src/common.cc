#include "common.h"
#include <ctime>
#include <cstdio>

std::map<std::string, pj> global_record;


void record(std::string key, int64_t value) {
    global_record[key] = pj(value);
}

void record(std::string key, double value) {
    global_record[key] = pj(value);
}


void record(std::string key, const std::string value) {
    global_record[key] = pj(value);
}


void record(std::string key, const std::vector<int> &values) {
    std::vector<pj> pj_vec;
    for (auto e : values) {
        pj_vec.push_back( pj((int64_t)e) );
    }
    global_record[key] = pj(pj_vec);
}


void record(std::string key, const std::vector<double> &values) {
    std::vector<pj> pj_vec;
    for (auto e : values) {
        pj_vec.push_back( pj((double)e) );
    }
    global_record[key] = pj(pj_vec);
}


void record(std::string key, const std::vector<pj> &values) {
    global_record[key] = pj(values);
}


void record(std::string key, const std::map<std::string, pj> &dict) {
    global_record[key] = pj(dict);
}


std::string global_record_to_string() {
    return pj(global_record).serialize( /* prettify = */ true );
}


// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string get_current_date_time(std::string format) {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), format.c_str(), &tstruct);

    return buf;
}
