#!/bin/sh
# fetch.sh DIR: download the quality gate's data (GATE.md) into DIR and check every file's sha256. The repository
# keeps only the sample's ids, the prompts and our outputs, never the sets' text.
#
#   QASPER v0.3 (AllenAI, CC BY 4.0): the test split with its official evaluator, and the train/dev tarball whose
#   dev file is the validation split of Hugging Face's allenai/qasper (its loading script at fdc9d82 names both
#   tarballs), used only to develop the prompts and the parser.
#   Re-DocRED (MIT): the test and dev splits at the repository's commit ccfb54f.
#   DocRED's 96 relation names (rel_info.json, MIT), from Hugging Face's thunlp/docred at 7985b4e.
#
# Re-running skips a file whose sha256 already checks; a file whose sha256 does not is an error, never kept.
set -eu

dir=${1:?usage: fetch.sh DIR}
mkdir -p "$dir"

if command -v sha256sum >/dev/null 2>&1; then
	sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
	sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

check() { # check FILE SHA256: fail unless FILE has that sha256
	got=$(sha256 "$1")
	if [ "$got" != "$2" ]; then
		echo "fetch.sh: $1 has sha256 $got, expected $2" >&2
		exit 1
	fi
}

download() { # download URL FILE SHA256
	if [ -f "$dir/$2" ] && [ "$(sha256 "$dir/$2")" = "$3" ]; then
		return
	fi
	curl -fsSL --retry 3 -o "$dir/$2.part" "$1"
	check "$dir/$2.part" "$3"
	mv "$dir/$2.part" "$dir/$2"
}

QASPER=https://qasper-dataset.s3.us-west-2.amazonaws.com
REDOCRED=https://raw.githubusercontent.com/tonytan48/Re-DocRED/ccfb54f5ddf5836027c87badda10f6dfc56efaac/data
DOCRED=https://huggingface.co/datasets/thunlp/docred/resolve/7985b4e0371e6c61a756feb41b7b27becf71c666/data

download "$QASPER/qasper-test-and-evaluator-v0.3.tgz" qasper-test-and-evaluator-v0.3.tgz \
	72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72
download "$QASPER/qasper-train-dev-v0.3.tgz" qasper-train-dev-v0.3.tgz \
	a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a
download "$REDOCRED/test_revised.json" redocred-test.json \
	ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c
download "$REDOCRED/dev_revised.json" redocred-dev.json \
	051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f
download "$DOCRED/rel_info.json.gz" rel_info.json.gz \
	7ef27efff537ba89ae66f6f4e60e4908d4df3860a4c2819ea94fa5ed696bdc70

tar -xzf "$dir/qasper-test-and-evaluator-v0.3.tgz" -C "$dir" \
	qasper-test-v0.3.json qasper_evaluator.py README-test.md
tar -xzf "$dir/qasper-train-dev-v0.3.tgz" -C "$dir" qasper-dev-v0.3.json
gzip -dc "$dir/rel_info.json.gz" >"$dir/rel_info.json"

check "$dir/qasper-test-v0.3.json" 6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5
check "$dir/qasper_evaluator.py" 781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe
check "$dir/README-test.md" 64616ee7ee72494afe5cd8296ab8d1f30b134c8a3aafa556d035ec7c9b97414c
check "$dir/qasper-dev-v0.3.json" 2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93
check "$dir/rel_info.json" 5ecf4e5e55c179fc83a3a3d19baa01efffecb26ba5edc0b4ac5a54ddf61fe3de
echo "fetch.sh: the quality gate's data is in $dir, every sha256 checked"
