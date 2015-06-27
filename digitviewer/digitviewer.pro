#-------------------------------------------------
#
# Project created by QtCreator 2015-06-25T06:06:16
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = digitviewer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    digitpane.cpp \
    digitdata.cpp

HEADERS  += mainwindow.h \
    digitpane.h \
    digitdata.h

FORMS    += mainwindow.ui

CONFIG += c++11

DISTFILES +=

RESOURCES += \
    resource.qrc
