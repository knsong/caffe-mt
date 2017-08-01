// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include <sys/stat.h>
using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, anydata, mtcnn} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
	"Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, vector<float> > > lines;
  std::string line, filename;
  size_t pos;

  bool isAnyData = FLAGS_backend == "anydata";
  bool isMTCNN = FLAGS_backend == "mtcnn";
  FLAGS_backend = "lmdb";

  while (std::getline(infile, line)){
	  float label;
	  std::istringstream iss(line);
	  iss >> filename;
	  std::vector<float> labels;
	  //printf("read label: ");
	  while (iss >> label){
		  labels.push_back(label);
		  //printf("%f ", label);
	  }

	  //label, roi_minx, roi_miny, roi_maxx, roi_maxy, pts
	  //if (isMTCNN)
	  //	CHECK_GE(labels.size(), 5) << "Incorrect label size " << filename;
	  lines.push_back(std::make_pair(filename, labels));
  }

  /*
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }
  */

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);

  Datum datum;
  Datum mtcnndatum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
	if (encoded && !enc.size() && !isAnyData) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
/*
	if (isAnyData)
		status = ReadAnyDataFileToDatum(root_folder + lines[line_id].first,
			lines[line_id].second, resize_height, resize_width, is_color,
			&datum);
	else 
  */  
    if (isMTCNN){
        /*    
		status = ReadImageToMTCNNDatum(root_folder + lines[line_id].first,
			lines[line_id].second, resize_height, resize_width, is_color,
			enc, &mtcnndatum);
        */
        if(string::npos == lines[line_id].first.find('.'))
            lines[line_id].first = lines[line_id].first+ ".jpg";
        status = ReadImageToDatum(root_folder + lines[line_id].first,
                        lines[line_id].second, resize_height, resize_width, is_color,
                        &mtcnndatum);    
    }
    else
        LOG(FATAL) << "Unsupported data format!";
/*
    else
		status = ReadImageToDatum(root_folder + lines[line_id].first,
			lines[line_id].second, resize_height, resize_width, is_color,
			enc, &datum);
*/            
    if (status == false) continue;
    if (check_size) {
		int channels = mtcnndatum.channels();
		int height =  mtcnndatum.height();
		int width =  mtcnndatum.width();
		const std::string& data = mtcnndatum.data();

		if (!data_size_initialized) {
			data_size = channels * height * width;
			data_size_initialized = true;
		} else {
			CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
			<< data.size();
		}
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
	if (isMTCNN)
		CHECK(mtcnndatum.SerializeToString(&out));
	else
		CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
