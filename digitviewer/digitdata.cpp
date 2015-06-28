#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <cassert>
#include "digitdata.h"

DigitData::DigitData(QObject *parent)
    : QObject(parent), m_pData(nullptr), m_pLabels(nullptr)
    , m_numberOfData(0), m_bLabel(false)
{
}

const digit_t & DigitData::getData(int idx) const
{
    assert(0 <= idx && idx < numberOfData());
    return (*m_pData)[idx];
}

char DigitData::getLabel(int idx) const
{
    assert(0 <= idx && idx < numberOfData());
    return (*m_pLabels)[idx];
}

// utitly function for performance
// convert ascii text to integer
static int myA2I(const char * begin, const char * end, bool * ok)
{
    int val = 0;
    for(; begin != end; ++begin) {
        if(*begin == '\r' || *begin == '\n')
            break;
        if(!(*ok = isdigit(*begin)))
            break;
        val = 10 * val + (*begin - '0');
    }

    return val;
}

// utitly function for performance
// make sure header column is valid
static bool colCheck(const char * begin, const char * end, int colno)
{
    const char pixel[] = {'p', 'i', 'x', 'e', 'l'};
    if(end - begin < sizeof(pixel))
        return false;
    if(!std::equal(pixel, pixel+sizeof(pixel), begin))
        return false;

    bool ok;
    int val = myA2I(begin + sizeof(pixel), end, &ok);

    return ok && (val == colno);
}

bool DigitData::setFile(const QString &filename)
{
    QFile f(filename);
    if(!f.open(QIODevice::ReadOnly))
        return false;

    const int bufsize = 800 * 10;
    std::array<char, bufsize> buf;
    qint64 lineLen = f.readLine(buf.data(), buf.size());
    if(lineLen == -1)
        return false;

    // make sure the header is correct
    const char label[] = { 'l','a','b','e','l',',' };
    if(lineLen < sizeof(label))
        return false;
    const bool bTrain = std::equal(label, label + sizeof(label), buf.data());

    auto prev = &buf[0];
    auto end = &buf[lineLen];
    if(bTrain)
        prev = std::find(prev, end, ',') + 1;
    decltype(prev) next;
    int colcount;
    for(colcount = 0; prev < end; prev = next + 1, ++colcount) {
        next = std::find(prev, end, ',');
        if(!colCheck(prev, next, colcount))
            break;
    }
    if(prev < end || colcount != DigitPixels)
        return false;

    // read digit data
    std::unique_ptr< std::vector<digit_t> > pNewData(new std::vector<digit_t>());
    std::unique_ptr< std::vector<char> > pNewLabels(new std::vector<char>());

    while((lineLen = f.readLine(buf.data(), buf.size())) != -1) {
        int val;
        bool ok;
        prev = &buf[0];
        end = &buf[lineLen];
        if(bTrain) {
            next = std::find(prev, end, ',');
            if(next - prev != 1 || !isdigit(*prev))
                return false;
            pNewLabels->push_back(*prev);
            prev = next + 1;
        }

        digit_t digit;
        for(colcount = 0; prev < end; prev = next + 1, ++colcount) {
            next = std::find(prev, end, ',');
            val = myA2I(prev, next, &ok);
            if(!ok || val < 0 || val > 255)
                break;
            digit[colcount] = val;
        }
        if(prev < end || colcount != DigitPixels)
            return false;
        pNewData->push_back(std::move(digit));
    }

    m_pData = std::move(pNewData);
    m_pLabels = std::move(pNewLabels);
    m_numberOfData = static_cast<int>(m_pData->size());
    m_bLabel = bTrain;

    emit dataChanged();

    return true;
}
