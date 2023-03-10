CONFIG += strict_c++ c++latest
CONFIG -= qt

TEMPLATE = app
CONFIG += console

mac* | linux* | freebsd {
	CONFIG(release, debug|release):CONFIG *= Release optimize_full
	CONFIG(debug, debug|release):CONFIG *= Debug
}

Release:OUTPUT_DIR=release
Debug:OUTPUT_DIR=debug

DESTDIR  = bin
OBJECTS_DIR = bin/build

win*{
	QMAKE_CXXFLAGS += /std:c++latest /permissive- /Zc:__cplusplus /Zc:char8_t

	QMAKE_CXXFLAGS += /MP /FS /diagnostics:caret
	QMAKE_CXXFLAGS += /wd4251
	QMAKE_CXXFLAGS_WARN_ON = /W4
	DEFINES += WIN32_LEAN_AND_MEAN NOMINMAX _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_WARNINGS

	QMAKE_CXXFLAGS_DEBUG -= -Zi
	QMAKE_CXXFLAGS_DEBUG *= /ZI
	Debug:QMAKE_LFLAGS += /DEBUG:FASTLINK /INCREMENTAL

	Release:QMAKE_CXXFLAGS += /Zi

	Release:QMAKE_LFLAGS += /DEBUG:FULL /OPT:REF /OPT:ICF /INCREMENTAL /TIME
}

linux*|mac*{
	QMAKE_CXXFLAGS += -std=c++2b

	QMAKE_CXXFLAGS_WARN_ON = -Wall -Wextra -Wdelete-non-virtual-dtor -Werror=duplicated-cond -Werror=duplicated-branches -Warith-conversion -Warray-bounds -Wattributes -Wcast-align -Wcast-qual -Wconversion -Wdate-time -Wduplicated-branches -Wendif-labels -Werror=overflow -Werror=return-type -Werror=shift-count-overflow -Werror=sign-promo -Werror=undef -Wextra -Winit-self -Wlogical-op -Wmissing-include-dirs -Wnull-dereference -Wpedantic -Wpointer-arith -Wredundant-decls -Wshadow -Wstrict-aliasing -Wstrict-aliasing=3 -Wuninitialized -Wunused-const-variable=2 -Wwrite-strings -Wlogical-op
	QMAKE_CXXFLAGS_WARN_ON += -Wno-missing-include-dirs -Wno-undef

	Release:DEFINES += NDEBUG=1
	Debug:DEFINES += _DEBUG
}

*g++*{
	QMAKE_CXXFLAGS += -fconcepts -ggdb3 -fuse-ld=gold
	#QMAKE_CXXFLAGS += -fsanitize=thread
	#QMAKE_LFLAGS += -fsanitize=thread
}

SOURCES += \
    src/main.cpp

HEADERS +=
