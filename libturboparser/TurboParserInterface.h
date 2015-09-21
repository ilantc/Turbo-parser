#include <stdio.h>
#include <stdlib.h>
#include "SequencePipe.h"
#include "DependencyPipe.h"

namespace TurboParserInterface {

class TurboTaggerWorker {
 public:
  TurboTaggerWorker();
  virtual ~TurboTaggerWorker();

  void LoadTaggerModel(const std::string &file_model);

  void Tag(const std::string &file_test,
           const std::string &file_prediction);

 private:
  SequenceOptions *tagger_options_;
  SequencePipe *tagger_pipe_;
};

class TurboParserWorker {
 public:
  TurboParserWorker();
  virtual ~TurboParserWorker();

  void LoadParserModel(const std::string &file_model);

  void Parse(const std::string &file_test,
             const std::string &file_prediction);

 private:
  DependencyOptions *parser_options_;
  DependencyPipe *parser_pipe_;
};

class TurboParserInterface {
 public:
  TurboParserInterface();
  virtual ~TurboParserInterface();

  void ClearArgumentList() {
    for (int i = 0; i < argc_; ++i) {
      if (argv_[i]) free(argv_[i]);
    }
    delete[] argv_;
    argc_ = 0;
  }

  void BuildArgumentList() {
    argc_ = 2;
    argv_ = new char*[argc_];
    argv_[0] = strdup("TurboParser");
    argv_[1] = strdup("--logtostderr");
  }

  TurboTaggerWorker *CreateTagger() {
    TurboTaggerWorker *tagger = new TurboTaggerWorker();
    taggers_.push_back(tagger);
    return tagger;
  }

  TurboParserWorker *CreateParser() {
    TurboParserWorker *parser = new TurboParserWorker();
    parsers_.push_back(parser);
    return parser;
  }

  void DeleteAllTaggers() {
    for (int i = 0; i < taggers_.size(); ++i) {
      delete taggers_[i];
    }
    taggers_.clear();
  }

  void DeleteAllParsers() {
    for (int i = 0; i < parsers_.size(); ++i) {
      delete parsers_[i];
    }
    parsers_.clear();
  }

 private:
  int argc_;
  char** argv_;
  vector<TurboTaggerWorker*> taggers_;
  vector<TurboParserWorker*> parsers_;
};

} // namespace TurboParserInterface.
