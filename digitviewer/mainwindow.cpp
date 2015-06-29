#include <QFileDialog>
#include <QUrl>
#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_data(new DigitData(this))
{
    ui->setupUi(this);

    ui->digit_pane->setEditGoTo(ui->editGoTo).setLabelUpTo(ui->labelUpTo).setData(m_data);
    connect(m_data, &DigitData::dataChanged, ui->digit_pane, &DigitPane::onDataChanged);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::onOpen);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::onOpen()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Digit data file"));
    if(filename.isNull())
        return;
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    if(m_data->setFile(filename)) {
        ui->labelTotalDigits->setText(QString::number(m_data->numberOfData()));
        QUrl fullpath(filename);
        setWindowTitle(QString(tr("Digit Viewer - %1")).arg(fullpath.fileName()));
        ui->statusBar->showMessage((m_data->isLabelAvailable() ? tr("Training data set") : tr("Test data set")));
    }
    QApplication::setOverrideCursor(QCursor(Qt::ArrowCursor));
}

void MainWindow::onOffsetEdited()
{
    ui->buttonPrevious->setFocus();
}
