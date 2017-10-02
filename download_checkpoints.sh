#!/bin/bash
#
# Scripts which download checkpoints
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir "${SCRIPT_DIR}/adv_inception_v3/"
mkdir "${SCRIPT_DIR}/ens3_adv_inception_v3_2017_08_18/"
mkdir "${SCRIPT_DIR}/ens4_adv_inception_v3_2017_08_18/"
mkdir "${SCRIPT_DIR}/ens_adv_inception_resnet_v2/"

wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz -C adv_inception_v3/
rm adv_inception_v3_2017_08_18.tar.gz

wget http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens3_adv_inception_v3_2017_08_18.tar.gz -C ens3_adv_inception_v3_2017_08_18/
rm ens3_adv_inception_v3_2017_08_18.tar.gz

wget http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz
tar -xvzf ens4_adv_inception_v3_2017_08_18.tar.gz -C ens4_adv_inception_v3_2017_08_18/
rm ens4_adv_inception_v3_2017_08_18.tar.gz

wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz -C ens_adv_inception_resnet_v2/
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
