#pragma once

#include <chrono>
#include <iostream>
#include <string>

class ElapseTimer
{
    std::chrono::system_clock::time_point start;
    std::string msg;

public:
    ElapseTimer(std::string start_message = "", bool show_construction_msg = false):
            start(std::chrono::high_resolution_clock::now())
    {
        if (show_construction_msg) std::cout << start_message << " timer started ..." << std::endl;
        msg = start_message;
    }

    double elapsed()
    {
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        return elapsed.count();
    }

    void printElapsed()
    {
        if (not msg.empty()) std::cout << msg << ": ";
        std::cout << "Time to execute: " << this->elapsed() << " seconds." << std::endl;
    }
};