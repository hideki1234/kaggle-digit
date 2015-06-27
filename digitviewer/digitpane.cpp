#include <QPainter>
#include <QStyleOption>
#include <QImage>
#include <QByteArray>

#include "digitpane.h"

DigitPane::DigitPane(QWidget *parent)
    : QWidget(parent), m_bLabels(false), m_dataOffset(0)
    , m_data(nullptr), m_editGoTo(nullptr), m_labelUpTo(nullptr)
{
    for(auto &raw_digit : m_digits)
        raw_digit.fill(0xFF);
}

DigitPane & DigitPane::setData(DigitData *d)
{
    m_data = d;
    setPage(0);

    return *this;
}

DigitPane & DigitPane::setEditGoTo(QLineEdit *e)
{
    m_editGoTo = e;
    return *this;
}

DigitPane & DigitPane::setLabelUpTo(QLabel *l)
{
    m_labelUpTo = l;
    return *this;
}

void DigitPane::paintEvent(QPaintEvent * /*e*/)
{
    QPainter dc(this);

    QStyleOption opt;
    opt.init(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &dc, this);

    for(int r = 0; r < rows; ++r)
        for(int c = 0; c < columns; ++c) {
            // convert 8-bit grayscale to 32 bit RGB
            const uchar * raw_bits = &m_digits[10 * r + c][0];
            std::array<uchar, DigitPixels*4> raw;
            for(int i = 0; i < DigitPixels; ++i) {
                const int j = 4*i;
                raw[j] = raw[j + 1] = raw[j + 2] = raw_bits[i];
                raw[j+3] = 0xff;
            }

            // draw the digit image
            QImage img(&raw[0], DigitWidth, DigitWidth, DigitWidth*sizeof(uint), QImage::Format_RGB32);
            dc.drawImage(coloffset + (coloffset + DigitWidth) * c, DigitWidth * r, img);
            if(m_bLabels) {
                char label[2];
                label[0] = m_labels[10 * r + c];
                label[1] = '\0';
                dc.drawText((coloffset+DigitWidth)*c, DigitWidth * r, coloffset, DigitWidth,
                            Qt::AlignVCenter | Qt::AlignRight, label);
            }
        }
}

void DigitPane::setPage(int newOffset)
{
    for(auto & digit : m_digits)
        digit.fill(0xFF);

    m_dataOffset = newOffset;
    int count = digitsPerPage;
    const int max = m_data->numberOfData();
    if(m_dataOffset + count > max)
        count = max - m_dataOffset;

    for(int i = 0; i < count; ++i) {
        m_digits[i] = m_data->getData(i + m_dataOffset);
    }

    update();
    showOffsets();}

void DigitPane::showOffsets()
{
    int upto = m_dataOffset + digitsPerPage - 1;
    const int max = m_data->numberOfData();
    if(upto >= max)
        upto = (max > 0 ? max - 1 : 0);

    m_editGoTo->setText(QString::number(m_dataOffset));
    m_labelUpTo->setText(QString::number(upto));
}

void DigitPane::onDataChanged()
{
    setPage(0);
}

void DigitPane::onOffsetEdited()
{
    onSetOffset(m_editGoTo->text());
}

void DigitPane::onSetOffset(QString newPageText)
{
    bool ok;
    int newPage = newPageText.toInt(&ok);
    if(ok)
        ok = (0 <= newPage) && (newPage < m_data->numberOfData());
    if(ok) {
        newPage = (newPage / digitsPerPage) * digitsPerPage;
        setPage(newPage);
    } else {
        showOffsets();
    }
}

void DigitPane::onNext()
{
    const int max = m_data->numberOfData();
    const int newOffset = m_dataOffset + digitsPerPage;
    if(newOffset < max)
        setPage(newOffset);
}

void DigitPane::onPrevious()
{
    const int newOffset = m_dataOffset - digitsPerPage;
    if(0 <= newOffset)
        setPage(newOffset);
}
