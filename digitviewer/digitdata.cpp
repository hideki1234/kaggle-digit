#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <cassert>
#include "digitdata.h"

DigitData::DigitData(QObject *parent)
    : QObject(parent), m_pData(nullptr), m_pLabels(nullptr), m_numberOfData(0)
{
}

const digit_t & DigitData::getData(int idx) const
{
    assert(0 <= idx && idx < numberOfData());
    return (*m_pData)[idx];
}

bool DigitData::setFile(const QString &filename)
{
    QFile f(filename);
    if(!f.open(QIODevice::ReadOnly))
        return false;

    QTextStream ts(&f);
    bool bTrain;

    // make sure the header
    QString row = ts.readLine();
    if(row.isNull())
        return false;

    const auto csv_delimiter = QString(",");
    auto columns = row.split(csv_delimiter);
    switch(columns.length()) {
    case 784:
        // test data
        bTrain = false;
        break;
    case 785:
        // training data
        bTrain = true;
        if(columns[0] != "label")
            return false;
        break;
    default:
        return false;
    }

    const auto ofs = (bTrain ? 1 : 0);
    for(int i = 0; i < DigitPixels; ++i) {
        const auto & col = columns[i+ofs];
        if(!col.startsWith("pixel"))
            return false;
        auto num_str = col.midRef(5);
        bool ok;
        if(num_str.toInt(&ok) != i || !ok)
            return false;
    }

    // read digit data
    std::unique_ptr< std::vector<digit_t> > pNewData(new std::vector<digit_t>());
    std::unique_ptr< std::vector<char> > pNewLabels(new std::vector<char>());

    for(int idx = 0; !ts.atEnd(); ++idx) {
        row = ts.readLine();
        columns = row.split(csv_delimiter);
        // make sure the number of columns is correct
        if(columns.length() != DigitPixels+ofs)
            return false;

        // take care of label if training data set
        if(bTrain) {
            if(columns[0].length() != 1 || !columns[0][0].isDigit())
                return false;
            pNewLabels->push_back(columns[0][0].toLatin1());
        }

        digit_t digit;
        for(int i = 0; i < DigitPixels; ++i) {
            bool ok;
            const int gray = columns[i+ofs].toInt(&ok);
            if(!ok || gray < 0 || gray > 255)
                return false;
            digit[i] = gray;
        }
        pNewData->push_back(std::move(digit));
    }

    m_pData = std::move(pNewData);
    m_pLabels = std::move(pNewLabels);
    m_numberOfData = static_cast<int>(m_pData->size());

    emit dataChanged();

    return true;
}
