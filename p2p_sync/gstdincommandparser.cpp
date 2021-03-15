#include "gstdincommandparser.h"
#include <QTimer>
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>

class GCommandSplitter final
{
public:
   explicit GCommandSplitter(const QString &input):
      input_{input}
   {

   }

   QStringList split() const
   {
      QStringList result;
      QString current;
      bool in_quotes{false};
      for (int32_t i = 0; i < input_.count(); ++i)
      {
         const auto c{input_[i]};
         if (c.isSpace())
         {
            if (in_quotes)
            {
               current += c;
            }
            else
            {
               if (!current.isEmpty())
               {
                  result.append(std::move(current));
                  current = QString();
               }
            }
         }
         else if (c == '"')
         {
            in_quotes = !in_quotes;
            current += c;
         }
         else
         {
            current += c;
         }
      }
      if (!current.isEmpty())
      {
         result.append(std::move(current));
      }
      return result;
   }

private:
   const QString input_;
};

GStdInCommandParser::GStdInCommandParser(GServer *server, QObject *parent) : QObject(parent), server_{server}
{

}

void GStdInCommandParser::executeCommand(const QString &cmd)
{
    if (cmd.startsWith("remote "))
    {
        auto params{GCommandSplitter(cmd.mid(7)).split()};
        if (params.size() > 0) {
            const auto proc{params[0]};
            params.pop_front();
            GCommandLineProcess cmd_proc{proc, params};
            server_->sendCommand(cmd_proc);
        }
    }
    else if (cmd.startsWith("send "))
    {
        auto params{GCommandSplitter(cmd.mid(5)).split()};
        for (auto path : params)
        {
            QFileInfo info{path.trimmed()};
            if (!info.exists())
            {
                continue;
            }
            if (info.isFile())
            {
                qDebug() << "sending file " << info.filePath();
                server_->sendCommand(GFileWrite{info.filePath()});
            }
            else if (info.isDir())
            {
                QDirIterator dir_it {path, QDirIterator::Subdirectories};
                while (dir_it.hasNext())
                {
                    auto it = dir_it.next();
                    if (QFileInfo(it).isFile())
                    {
                       qDebug() << "sending file " << it;
                       server_->sendCommand(GFileWrite{it});
                    }
                }
            }
        }
    }
    else
    {
        auto params{cmd.split(' ')};
        if (params.size() > 0) {
            const auto proc{params[0]};
            params.pop_front();
            GCommandLineProcess * cmd_proc = new GCommandLineProcess{proc, params, this};
            QObject::connect(cmd_proc, &GCommand::finished, cmd_proc, &QObject::deleteLater);
            QObject::connect(cmd_proc, &GCommand::outputAvailable, cmd_proc, [](auto out) {
                qDebug().noquote() << out;
            });
            QTimer::singleShot(0, cmd_proc, [cmd_proc]() {
                cmd_proc->execute();
            });
        }
    }
}
