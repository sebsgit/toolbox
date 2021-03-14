#ifndef GCOMMANDSERIALIZER_H
#define GCOMMANDSERIALIZER_H

#include <QObject>
#include <QByteArray>

#include <memory>
#include <array>

using GCommandTag = std::array<uint8_t, 8>;
using GCommandSize = int64_t;

struct GCommandHeader
{
    static constexpr size_t SIZE{64};

    std::array<uint8_t, SIZE> data{};

    bool isValid() const noexcept;

    void writeTag(const GCommandTag &tag) noexcept;
    GCommandTag readTag() const noexcept;

    void writeSize(const GCommandSize &size) noexcept;
    GCommandSize readSize() const noexcept;

    QByteArray toByteArray() const;
};

struct GEncodedCommand
{
    GCommandHeader header;
    QByteArray data;
};

class GCommand : public QObject
{
    Q_OBJECT
public:
    using QObject::QObject;

    virtual GEncodedCommand serialize() const = 0;
    virtual void execute() = 0;
    virtual QString description() const = 0;

signals:
    void outputAvailable(const QByteArray &data);
    void finished();
};

class GCommandLineProcess : public GCommand
{
    Q_OBJECT
public:
    GCommandLineProcess(const QString &prog, const QStringList &args = {}, QObject * parent = nullptr);
    ~GCommandLineProcess() override;
    GEncodedCommand serialize() const override;
    void execute() override;
    QString description() const override;

    static std::unique_ptr<GCommandLineProcess> fromData(const GEncodedCommand &raw_data);

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

class GFileWrite : public GCommand
{
    Q_OBJECT
public:
    GFileWrite(const QString &path, const QByteArray &compressed_content, QObject * parent = nullptr);
    GFileWrite(const QString &path, QObject * parent = nullptr);
    GEncodedCommand serialize() const override;
    void execute() override;
    QString description() const override;

    static std::unique_ptr<GFileWrite> fromData(const GEncodedCommand &raw_data);

private:
    QString path_;
    QByteArray content_;
};

class GConsoleOutput : public GCommand
{
    Q_OBJECT
public:
    GConsoleOutput(const QString &what, QObject * parent = nullptr);
    GEncodedCommand serialize() const override;
    void execute() override;
    QString description() const override;

    static std::unique_ptr<GConsoleOutput> fromData(const GEncodedCommand &raw_data);

private:
    QString what_;
};

class GCommandSerializer : public QObject
{
    Q_OBJECT
public:
    explicit GCommandSerializer(QObject *parent = nullptr);
    ~GCommandSerializer();

    void addData(const QByteArray &data);

    GCommand* pendingCommand();

    static QByteArray version();

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // GCOMMANDSERIALIZER_H
