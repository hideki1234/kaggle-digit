#ifndef DIGITPANE_H
#define DIGITPANE_H

#include <QWidget>
#include <QLineEdit>
#include <QLabel>
#include <array>
#include "digitdata.h"

class DigitPane : public QWidget
{
    Q_OBJECT

private:
    const static int columns = 10;
    const static int rows = 10;
    const static int coloffset = 12;
    const static int digitsPerPage = columns * rows;

    std::array< std::array<uchar, DigitPixels>, digitsPerPage> m_digits;
    std::array<char, digitsPerPage> m_labels;
    bool m_bLabels;
    int m_dataOffset;

    DigitData * m_data;
    QLineEdit * m_editGoTo;
    QLabel * m_labelUpTo;

    void setPage(int newOffset);
    void showOffsets();

public:
    explicit DigitPane(QWidget *parent = 0);
    DigitPane & setData(DigitData * d);
    DigitPane & setEditGoTo(QLineEdit * e);
    DigitPane & setLabelUpTo(QLabel * l);

signals:

public slots:
    void onDataChanged();
    void onOffsetEdited();
    void onSetOffset(QString newPageText);
    void onNext();
    void onPrevious();

protected:
    void paintEvent(QPaintEvent * e) override;
};

#endif // DIGITPANE_H
