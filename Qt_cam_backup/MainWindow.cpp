#include "MainWindow.h"
#include "AppSettings.h"
#include "DataStorageConfig.h"
#include "SourceConfigWidget.h"
#include "dataSinks/DataSinks.h"
#include "dataSources/DataProducer.h"
#include "ui_MainWindow.h"

#include <QDebug>
#include <QState>
#include <QStateMachine>
#include <QTime>

class MainWindow::Priv {
public:
    Ui::MainWindow ui;
    AppSettings settings;
    QStateMachine uiStates;
    DataProducer producer;
    DataSinks sink;
};

static void setLayoutEnabled(QLayout* layout, bool enabled)
{
    for (int i = 0; i < layout->count(); ++i) {
        if (auto widget = layout->itemAt(i)->widget())
            widget->setEnabled(enabled);
    }
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , priv_(new Priv())
{
    priv_->ui.setupUi(this);

    QObject::connect(&priv_->producer, &DataProducer::error, this, &MainWindow::hanleDebugMessage);
    QObject::connect(&priv_->sink, &DataSinks::statusMessage, this, &MainWindow::hanleDebugMessage);

    QState* sourceSelectionState = new QState(&priv_->uiStates);
    QObject::connect(sourceSelectionState, &QState::entered, [this]() {
        this->setAsCentral(new SourceConfigWidget(priv_->settings));
        priv_->ui.mainButton->setText(tr("Start"));
    });

    QState* targetSettingsState = new QState(&priv_->uiStates);

    //TODO: add program-wide PIN code to hide settings
    //TODO: add status / messages window
    QObject::connect(targetSettingsState, &QState::entered, [this]() {
        this->setAsCentral(new DataStorageConfig(priv_->settings));
        priv_->ui.mainButton->setText(tr("Back to main window"));
        priv_->ui.settingsButton->setEnabled(false);
    });
    QObject::connect(targetSettingsState, &QState::exited, [this]() {
        priv_->ui.settingsButton->setEnabled(true);
    });

    QState* configureSourcesState = new QState(&priv_->uiStates);
    QObject::connect(configureSourcesState, &QState::entered, [this]() {
        priv_->producer.configure(priv_->settings.currentSourceSettings());
    });
    QState* configureSinksState = new QState(&priv_->uiStates);
    QObject::connect(configureSinksState, &QState::entered, [this]() {
        priv_->sink.configure(priv_->settings.currentTargetSettings());
    });

    QState* sendingState = new QState(&priv_->uiStates);
    QObject::connect(sendingState, &QState::entered, [this]() {
        priv_->producer.start();
        setLayoutEnabled(priv_->ui.centralLayout, false);
        priv_->ui.mainButton->setText(tr("Stop"));
        priv_->ui.settingsButton->setEnabled(false);
    });
    QObject::connect(sendingState, &QState::exited, [this]() {
        priv_->producer.stop();
        priv_->sink.finalize();
        setLayoutEnabled(priv_->ui.centralLayout, true);
        priv_->ui.settingsButton->setEnabled(true);
    });

    sourceSelectionState->addTransition(priv_->ui.mainButton, &QPushButton::clicked, configureSourcesState);
    configureSourcesState->addTransition(&priv_->producer, &DataProducer::configureDone, configureSinksState);
    configureSinksState->addTransition(&priv_->sink, &DataSinks::configureDone, sendingState);
    sendingState->addTransition(priv_->ui.mainButton, &QPushButton::clicked, sourceSelectionState);
    //TODO: finalizing state
    sourceSelectionState->addTransition(priv_->ui.settingsButton, &QPushButton::clicked, targetSettingsState);
    targetSettingsState->addTransition(priv_->ui.mainButton, &QPushButton::clicked, sourceSelectionState);
    //TODO: error state

    QObject::connect(&priv_->producer, &DataProducer::error, [](auto errorString) {
        qDebug() << errorString;
    });
    QObject::connect(&priv_->producer, &DataProducer::dataAvailable, &priv_->sink, &DataSinks::processData);

    priv_->uiStates.setInitialState(sourceSelectionState);
    priv_->uiStates.start();
}

MainWindow::~MainWindow() = default;

void MainWindow::setAsCentral(QWidget* widget)
{
    while (auto child = priv_->ui.centralLayout->takeAt(0)) {
        delete child->widget();
        delete child;
    }
    priv_->ui.centralLayout->addWidget(widget);
}

void MainWindow::hanleDebugMessage(const QString& message)
{
    const int maxMessages { 300 };
    if (priv_->ui.listWidget_messages->count() > maxMessages) {
        while (priv_->ui.listWidget_messages->count() > maxMessages / 2)
            delete priv_->ui.listWidget_messages->takeItem(0);
    }
    priv_->ui.listWidget_messages->addItem(QTime::currentTime().toString() + ": " + message);
    priv_->ui.listWidget_messages->scrollToBottom();
}
