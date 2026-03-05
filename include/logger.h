#pragma once

#include <cstdio>
#include <ctime>
#include <string>
#include <cstdarg>

enum class LogLevel { DEBUG = 0, INFO, WARN, ERROR };


#define LOG_RESET  "\033[0m"
#define LOG_CYAN   "\033[36m"
#define LOG_GREEN  "\033[32m"
#define LOG_YELLOW "\033[33m"
#define LOG_RED    "\033[31m"

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void set_level(LogLevel lvl) { min_level_ = lvl; }

    void open_file(const char* path) {
        if (file_) fclose(file_);
        file_ = fopen(path, "w");
    }

    ~Logger() { if (file_) fclose(file_); }

   
    void log(LogLevel lvl, const char* fmt, ...) {
        if (lvl < min_level_) return;

        char ts[20];
        time_t now = time(nullptr);
        strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));

        const char* label  = label_of(lvl);
        const char* color  = color_of(lvl);

       
        char msg[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);

        if (file_)
            fprintf(file_, "[%s] [%s] %s\n", ts, label, msg);
    }

  
    void debug(const char* fmt, ...) { va_forward(LogLevel::DEBUG, fmt); }
    void info (const char* fmt, ...) { va_forward(LogLevel::INFO,  fmt); }
    void warn (const char* fmt, ...) { va_forward(LogLevel::WARN,  fmt); }
    void error(const char* fmt, ...) { va_forward(LogLevel::ERROR, fmt); }

private:
    Logger() : min_level_(LogLevel::DEBUG), file_(nullptr) {}
    FILE*    file_;
    LogLevel min_level_;

    static const char* label_of(LogLevel l) {
        switch(l) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
        }
        return "?????";
    }
    static const char* color_of(LogLevel l) {
        switch(l) {
            case LogLevel::DEBUG: return LOG_CYAN;
            case LogLevel::INFO:  return LOG_GREEN;
            case LogLevel::WARN:  return LOG_YELLOW;
            case LogLevel::ERROR: return LOG_RED;
        }
        return LOG_RESET;
    }

    
    void va_forward(LogLevel lvl, const char* fmt, ...) {
        if (lvl < min_level_) return;
        char msg[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        log(lvl, "%s", msg);
    }
};

#define LOG_DEBUG(...) Logger::instance().log(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  Logger::instance().log(LogLevel::INFO,  __VA_ARGS__)
#define LOG_WARN(...)  Logger::instance().log(LogLevel::WARN,  __VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().log(LogLevel::ERROR, __VA_ARGS__)