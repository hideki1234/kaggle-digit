#ifndef DIGITDATA_H
#define DIGITDATA_H

#include <QObject>
#include <memory>
#include <vector>
#include <array>

const int DigitWidth = 28;
const int DigitPixels = DigitWidth*DigitWidth;
typedef std::array<uchar, DigitPixels> digit_t;

class DigitData : public QObject
{
    Q_OBJECT

private:
    std::unique_ptr< std::vector<digit_t> > m_pData;
    std::unique_ptr< std::vector<char> > m_pLabels;
    int m_numberOfData;
    bool m_bLabel;

public:
    explicit DigitData(QObject *parent = 0);

    int numberOfData() const {return m_numberOfData;}
    const digit_t & getData(int idx) const;
    bool isLabelAvailable() const {return m_bLabel;}
    char getLabel(int idx) const;
    bool setFile(const QString & filename);
        // true: succeeded; false: failed

signals:
    void dataChanged();

public slots:
};

#endif // DIGITDATA_H
