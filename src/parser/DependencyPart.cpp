// Copyright (c) 2012-2013 Andre Martins
// All Rights Reserved.
//
// This file is part of TurboParser 2.1.
//
// TurboParser 2.1 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TurboParser 2.1 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with TurboParser 2.1.  If not, see <http://www.gnu.org/licenses/>.

#include "DependencyPart.h"

void DependencyParts::DeleteAll() {
  for (int i = 0; i < NUM_DEPENDENCYPARTS; ++i) {
    offsets_[i] = -1;
  }

  DeleteIndices();

  for (iterator iter = begin(); iter != end(); iter++) {
    if  ((*iter) != NULL) {
      delete (*iter);
      *iter = NULL;
    }
  }

  clear();
}

void DependencyParts::DeleteIndices() {
  for (int i = 0; i < index_.size(); ++i) {
    index_[i].clear();
  }
  index_.clear();

  for (int i = 0; i < index_labeled_.size(); ++i) {
    for (int j = 0; j < index_labeled_[i].size(); ++j) {
      index_labeled_[i][j].clear();
    }
    index_labeled_[i].clear();
  }
  index_labeled_.clear();
}

void DependencyParts::BuildIndices(int sentence_length, bool labeled) {
  DeleteIndices();
  index_.resize(sentence_length);
  for (int h = 0; h < sentence_length; ++h) {
    index_[h].resize(sentence_length);
    for (int m = 0; m < sentence_length; ++m) {
      index_[h][m] = -1;
    }
  }

  int offset, num_basic_parts;
  GetOffsetArc(&offset, &num_basic_parts);
  for (int r = 0; r < num_basic_parts; ++r) {
    Part *part = (*this)[offset + r];
    CHECK(part->type() == DEPENDENCYPART_ARC);
    int h = static_cast<DependencyPartArc*>(part)->head();
    int m = static_cast<DependencyPartArc*>(part)->modifier();
    index_[h][m] = offset + r;
  }

  if (labeled) {
    index_labeled_.resize(sentence_length);
    for (int h = 0; h < sentence_length; ++h) {
      index_labeled_[h].resize(sentence_length);
      for (int m = 0; m < sentence_length; ++m) {
        index_labeled_[h][m].clear();
      }
    }

    int offset, num_labeled_arcs;
    GetOffsetLabeledArc(&offset, &num_labeled_arcs);
    for (int r = 0; r < num_labeled_arcs; ++r) {
      Part *part = (*this)[offset + r];
      CHECK(part->type() == DEPENDENCYPART_LABELEDARC);
      int h = static_cast<DependencyPartLabeledArc*>(part)->head();
      int m = static_cast<DependencyPartLabeledArc*>(part)->modifier();
      index_labeled_[h][m].push_back(offset + r);
    }
  }
}

void DependencyParts::BuildEdgePartMapping(vector<vector<int> > *edge2parts) {
	int offset, num_arcs;
	GetOffsetArc(&offset, &num_arcs);
	(*edge2parts).assign(num_arcs,vector<int>());

//	LOG(INFO) << "n_parts = " << (*this).size();
//	LOG(INFO) << "n_arc = " << num_arcs;
	int currOffset,curr_n,r;
	GetOffsetSibl(&currOffset,&curr_n);
//	LOG(INFO) << "n_sibs = " << curr_n << ", offset sibs = " << currOffset;
	for (r = currOffset; r < curr_n + currOffset; ++r) {
	     DependencyPartSibl *part = static_cast<DependencyPartSibl*>((*this)[r]);
	     int r1 = FindArc(part->head(), part->modifier());
	     int r2 = FindArc(part->head(), part->sibling());
	     (*edge2parts)[r1].push_back(r);
	     (*edge2parts)[r2].push_back(r);
	}

	GetOffsetGrandpar(&currOffset,&curr_n);
//	LOG(INFO) << "n_gp = " << curr_n << ", offset gp = " << currOffset;
	for (r = currOffset; r < curr_n + currOffset; ++r) {
		DependencyPartGrandpar *part = static_cast<DependencyPartGrandpar*>((*this)[r]);
		int r1 = FindArc(part->grandparent(), part->head());
		int r2 = FindArc(part->head(), part->modifier());
		(*edge2parts)[r1].push_back(r);
		(*edge2parts)[r2].push_back(r);
	}
	GetOffsetGrandSibl(&currOffset,&curr_n);
	//	LOG(INFO) << "n_gp = " << curr_n << ", offset gp = " << currOffset;
	for (r = currOffset; r < curr_n + currOffset; ++r) {
		DependencyPartGrandSibl *part = static_cast<DependencyPartGrandSibl*>((*this)[r]);
		int r1 = FindArc(part->grandparent(), part->head());
		int r2 = FindArc(part->head(), part->modifier());
		int r3 = FindArc(part->head(), part->sibling());
		(*edge2parts)[r1].push_back(r);
		(*edge2parts)[r2].push_back(r);
		(*edge2parts)[r3].push_back(r);
	}
	GetOffsetTriSibl(&currOffset,&curr_n);
	//	LOG(INFO) << "n_gp = " << curr_n << ", offset gp = " << currOffset;
	for (r = currOffset; r < curr_n + currOffset; ++r) {
		DependencyPartTriSibl *part = static_cast<DependencyPartTriSibl*>((*this)[r]);
		int r1 = FindArc(part->head(), part->modifier());
		int r2 = FindArc(part->head(), part->sibling());
		int r3 = FindArc(part->head(), part->other_sibling());
		(*edge2parts)[r1].push_back(r);
		(*edge2parts)[r2].push_back(r);
		(*edge2parts)[r3].push_back(r);
	}
}
