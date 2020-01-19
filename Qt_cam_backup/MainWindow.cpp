#include "MainWindow.h"
#include "AppSettings.h"
#include "DataStorageConfig.h"
#include "PINSelectionWidget.h"
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
    Ui::MainWindow ui = {};
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

    auto pinSelectionState = new QState(&priv_->uiStates);
    auto sourceSelectionState = new QState(&priv_->uiStates);

    QObject::connect(pinSelectionState, &QState::entered, [this, pinSelectionState, sourceSelectionState]() {
        auto pinSelector = new PINSelectionWidget(priv_->settings);
        this->setAsCentral(pinSelector);
        pinSelectionState->addTransition(pinSelector, &PINSelectionWidget::validPINEntered, sourceSelectionState);
        QObject::connect(pinSelector, &PINSelectionWidget::statusMessage, this, &MainWindow::hanleDebugMessage);
        priv_->ui.mainButton->setEnabled(false);
        priv_->ui.settingsButton->setEnabled(false);
    });

    QObject::connect(sourceSelectionState, &QState::entered, [this]() {
        this->setAsCentral(new SourceConfigWidget(priv_->settings));
        priv_->ui.mainButton->setText(tr("Start"));
        priv_->ui.mainButton->setEnabled(true);
        priv_->ui.settingsButton->setEnabled(true);
    });

    auto targetSettingsState = new QState(&priv_->uiStates);
    QObject::connect(targetSettingsState, &QState::entered, [this]() {
        this->setAsCentral(new DataStorageConfig(priv_->settings));
        priv_->ui.mainButton->setText(tr("Back to main window"));
        priv_->ui.settingsButton->setEnabled(false);
    });
    QObject::connect(targetSettingsState, &QState::exited, [this]() {
        priv_->ui.settingsButton->setEnabled(true);
        priv_->ui.mainButton->setEnabled(true);
    });

    auto configureSourcesState = new QState(&priv_->uiStates);
    QObject::connect(configureSourcesState, &QState::entered, [this]() {
        priv_->producer.configure(priv_->settings.currentSourceSettings());
    });
    auto configureSinksState = new QState(&priv_->uiStates);
    QObject::connect(configureSinksState, &QState::entered, [this]() {
        priv_->sink.configure(priv_->settings.currentTargetSettings());
    });

    auto sendingState = new QState(&priv_->uiStates);
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

    priv_->uiStates.setInitialState(pinSelectionState);
    priv_->uiStates.start();
}

MainWindow::~MainWindow() = default;

void MainWindow::setAsCentral(QWidget* widget)
{
    while (auto child = priv_->ui.centralLayout->takeAt(0)) {
        if (child->widget()) {
            child->widget()->deleteLater();
        }
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
