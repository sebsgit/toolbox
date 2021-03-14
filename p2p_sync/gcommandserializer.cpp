#include "gcommandserializer.h"

#include <QProcess>
#include <QtEndian>
#include <QDataStream>
#include <QBuffer>
#include <QQueue>
#include <QtDebug>
#include <QFile>
#include <QDir>
#include <cstring>
#include <memory>
#include <QtGlobal>

static constexpr quint32 GSERIALIZER_VERSION{0x000001};
static constexpr size_t GHEADER_TAG_OFFSET{4};
static constexpr size_t GHEADER_SIZE_OFFSET{GCommandTag{}.size() + GHEADER_TAG_OFFSET};

bool GCommandHeader::isValid() const noexcept
{
    return std::memcmp("gcmd", data.data(), 4) == 0;
}

void GCommandHeader::writeTag(const GCommandTag &tag) noexcept
{
    std::memcpy(data.data(), "gcmd", 4);
    std::memcpy(data.data() + GHEADER_TAG_OFFSET, tag.data(), tag.size());
}

GCommandTag GCommandHeader::readTag() const noexcept
{
    GCommandTag tag;
    std::memcpy(tag.data(), data.data() + GHEADER_TAG_OFFSET, tag.size());
    return tag;
}

void GCommandHeader::writeSize(const GCommandSize &size) noexcept
{
    const auto transfer_format{qToLittleEndian(size)};
    std::memcpy(data.data() + GHEADER_SIZE_OFFSET, &transfer_format, sizeof(transfer_format));
}

GCommandSize GCommandHeader::readSize() const noexcept
{
    GCommandSize transfer_format{0};
    std::memcpy(&transfer_format, data.data() + GHEADER_SIZE_OFFSET, sizeof(transfer_format));
    return qFromLittleEndian(transfer_format);
}

QByteArray GCommandHeader::toByteArray() const
{
    QByteArray result;
    result.resize(SIZE);
    std::memcpy(result.data(), data.data(), SIZE);
    return result;
}

class GCommandLineProcess::Priv
{
public:
    QString program;
    QStringList arguments;
    QProcess proc;
};

static constexpr GCommandTag GCMD_LINE_PROC_TAG{{'c','m','d','l','i','n','e','0'}};
static constexpr GCommandTag GFILE_WRITE_TAG{{'f','i','l','w','r','i','t','0'}};
static constexpr GCommandTag GCONSOLE_PRINT_TAG{{'p','r','i','n','t','c','m','d'}};

GCommandLineProcess::GCommandLineProcess(const QString &prog, const QStringList &args, QObject * parent):
    GCommand(parent),
    priv_{std::make_unique<Priv>()}
{
    priv_->program = prog;
    priv_->arguments = args;
    priv_->proc.setProgram(prog);
    priv_->proc.setArguments(args);
    QObject::connect(&priv_->proc, &QProcess::readyReadStandardError, this, [this]()
     {
         emit outputAvailable(priv_->proc.readAllStandardError());
     });
    QObject::connect(&priv_->proc, &QProcess::readyReadStandardOutput, this, [this]()
     {
         emit outputAvailable(priv_->proc.readAllStandardOutput());
     });
    QObject::connect(&priv_->proc, &QProcess::errorOccurred, this, [this]()
     {
        emit outputAvailable((priv_->program + ": " + priv_->proc.errorString()).toLocal8Bit());
     });
    QObject::connect(&priv_->proc, SIGNAL(finished(int, QProcess::ExitStatus)), this, SIGNAL(finished()));
}

GCommandLineProcess::~GCommandLineProcess() = default;

GEncodedCommand GCommandLineProcess::serialize() const
{
    GEncodedCommand result;

    QBuffer buff{&result.data};
    buff.open(QIODevice::WriteOnly);
    QDataStream stream{&buff};
    stream << priv_->program;
    for (auto & arg : priv_->arguments)
    {
        stream << arg;
    }

    result.header.writeTag(GCMD_LINE_PROC_TAG);
    result.header.writeSize(result.data.size());
    return result;
}

std::unique_ptr<GCommandLineProcess> GCommandLineProcess::fromData(const GEncodedCommand &raw_data)
{
    if (raw_data.header.readTag() == GCMD_LINE_PROC_TAG)
    {
        if (raw_data.header.readSize() == raw_data.data.size())
        {
            QString command;
            QStringList arguments;
            QBuffer buff;
            buff.setData(raw_data.data);
            buff.open(QIODevice::ReadOnly);
            QDataStream stream{&buff};
            stream >> command;
            while (!stream.atEnd())
            {
                QString arg;
                stream >> arg;
                arguments.append(std::move(arg));
            }
            return std::make_unique<GCommandLineProcess>(command, arguments);
        }
    }
    return nullptr;
}

void GCommandLineProcess::execute()
{
    if (!priv_->proc.isOpen())
    {
        priv_->proc.start();
        priv_->proc.closeWriteChannel();
    }
}

QString GCommandLineProcess::description() const
{
    return "Process execute: " + priv_->program;
}

GFileWrite::GFileWrite(const QString &path, const QByteArray &compressed_content, QObject * parent): GCommand(parent),
    path_{path},
    content_{compressed_content}
{}

GFileWrite::GFileWrite(const QString &path, QObject * parent): GCommand(parent),
    path_{path}
{
    QFile file{path};
    if (file.open(QIODevice::ReadOnly))
    {
        content_ = qCompress(file.readAll());
    }
}

void GFileWrite::execute()
{
    const auto clean_path{QDir::cleanPath(path_)};
    const auto last_sep{clean_path.lastIndexOf('/')};
    if (last_sep != -1)
    {
        const auto paths{clean_path.mid(0, last_sep)};
        auto curr{QDir::current()};
        curr.mkpath(paths);
    }
    QFile file{clean_path};
    if (file.open(QIODevice::WriteOnly))
    {
        file.write(qUncompress(content_));
    }

    emit finished();
}

GEncodedCommand GFileWrite::serialize() const
{
    GEncodedCommand result;

    {
        QBuffer buff{&result.data};
        buff.open(QIODevice::WriteOnly);
        QDataStream stream{&buff};
        stream << path_;
        stream << content_;
    }

    result.header.writeTag(GFILE_WRITE_TAG);
    result.header.writeSize(result.data.size());
    return result;
}

QString GFileWrite::description() const
{
    return "File write, target: " + path_;
}

std::unique_ptr<GFileWrite> GFileWrite::fromData(const GEncodedCommand &raw_data)
{
    if (raw_data.header.readTag() == GFILE_WRITE_TAG)
    {
        if (raw_data.header.readSize() == raw_data.data.size())
        {
            QString path;
            QByteArray content;
            QBuffer buff;
            buff.setData(raw_data.data);
            buff.open(QIODevice::ReadOnly);
            QDataStream stream{&buff};
            stream >> path;
            stream >> content;
            return std::make_unique<GFileWrite>(path, content);
        }
    }
    return nullptr;
}

GConsoleOutput::GConsoleOutput(const QString &what, QObject * parent) : GCommand(parent),
    what_{what}
{}

GEncodedCommand GConsoleOutput::serialize() const
{
    GEncodedCommand result;

    {
        QBuffer buff{&result.data};
        buff.open(QIODevice::WriteOnly);
        QDataStream stream{&buff};
        stream << what_;
    }

    result.header.writeTag(GCONSOLE_PRINT_TAG);
    result.header.writeSize(result.data.size());
    return result;
}

void GConsoleOutput::execute()
{
    qDebug().noquote() << what_;
    emit finished();
}

QString GConsoleOutput::description() const
{
    return "console print";
}

std::unique_ptr<GConsoleOutput> GConsoleOutput::fromData(const GEncodedCommand &raw_data)
{
    if (raw_data.header.readTag() == GCONSOLE_PRINT_TAG)
    {
        if (raw_data.header.readSize() == raw_data.data.size())
        {
            QString what;
            QBuffer buff;
            buff.setData(raw_data.data);
            buff.open(QIODevice::ReadOnly);
            QDataStream stream{&buff};
            stream >> what;
            return std::make_unique<GConsoleOutput>(what);
        }
    }
    return nullptr;
}

class GCommandSerializer::Priv
{
public:
    QByteArray buffer;
    QQueue<GCommand*> commands;
};

GCommandSerializer::GCommandSerializer(QObject *parent) : QObject(parent),
    priv_{std::make_unique<Priv>()}
{

}

GCommandSerializer::~GCommandSerializer() = default;

static std::unique_ptr<GCommand> parseOne(const QByteArray &buffer, int64_t &bytesRead)
{
    if (buffer.size() > GCommandHeader::SIZE)
    {
        for (size_t i = 0; i < buffer.size() - GCommandHeader::SIZE; ++i)
        {
            GCommandHeader header;
            std::memcpy(header.data.data(), buffer.data() + i, GCommandHeader::SIZE);
            if (header.isValid())
            {
                if (buffer.size() >= header.readSize() + GCommandHeader::SIZE + i)
                {
                    GEncodedCommand raw_data;
                    raw_data.header = header;
                    raw_data.data = buffer.mid(GCommandHeader::SIZE + i, header.readSize());
                    if (auto cmd = GCommandLineProcess::fromData(raw_data))
                    {
                        bytesRead = GCommandHeader::SIZE + header.readSize() + i;
                        return cmd;
                    }
                    else if (auto cmd = GFileWrite::fromData(raw_data))
                    {
                        bytesRead = GCommandHeader::SIZE + header.readSize() + i;
                        return cmd;
                    }
                    else if (auto cmd = GConsoleOutput::fromData(raw_data))
                    {
                        bytesRead = GCommandHeader::SIZE + header.readSize() + i;
                        return cmd;
                    }
                }
            }
        }
    }
    return nullptr;
}

void GCommandSerializer::addData(const QByteArray &data)
{
    priv_->buffer += data;

    int64_t bytes_to_remove{0};
    while (auto cmd = parseOne(priv_->buffer, bytes_to_remove))
    {
        cmd->setParent(this);
        priv_->commands.push_back(cmd.release());
        priv_->buffer.remove(0, bytes_to_remove);
    }
}

GCommand *GCommandSerializer::pendingCommand()
{
    if (!priv_->commands.isEmpty())
    {
        auto result{std::move(priv_->commands.front())};
        priv_->commands.pop_front();
        return result;
    }

    return nullptr;
}

QByteArray GCommandSerializer::version()
{
    QByteArray ver;
    ver.resize(sizeof(GSERIALIZER_VERSION));
    std::memcpy(ver.data(), &GSERIALIZER_VERSION, ver.size());
    return ver;
}
