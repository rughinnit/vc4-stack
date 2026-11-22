#!/bin/sh
# fetch the three non-classification graphs used by docs/MODELS.md.
#
# The ONNX Model Zoo stores model files in Git LFS; raw.githubusercontent.com
# returns the 130-byte pointer, not the graph, so the media host is used.
# Sizes and SHA-256 are the files the page's numbers were measured on.
set -eu
cd "$(dirname "$0")"
mkdir -p zoo
fetch() {
	url=$1; out=zoo/$2; sum=$3
	[ -f "$out" ] || curl -fL --max-time 600 -o "$out" "$url"
	echo "$sum  $out" | sha256sum -c -
}
fetch "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx" \
	super-resolution-10.onnx 85f36ff88cc504a24af5e0602148bc56a8aa09a58eca8c0da2756f3e8186035e
fetch "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/style_transfer/fast_neural_style/model/candy-8.onnx" \
	candy-8.onnx e3c6f442e12014c5fec12b4307c14ca303df185cda19f67522afeef40c4d22b8
fetch "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12-int8.onnx" \
	fcn-resnet50-12-int8.onnx 67fcba8ebd435c2fb6b297d2e7c084f3c3b9ae6bc7ae41453b748ae4ed8c0331
