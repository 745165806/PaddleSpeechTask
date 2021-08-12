#!/bin/bash
CURR_ROOT=${PWD}
echo 'start to get iwslt2012 dataset!'

if [ ! -f '2012-03.tgz' ] ; then
$(wget -c 'https://drive.google.com/file/d/1aTW5gG2xCZbfNy5rOzG7iSJ7TVGywxcx/view?usp=sharing') || exit -1
echo 'download 2012-03.tgz'
fi

if [ ! -f '2012-03-test.tgz' ] ; then
$(wget -c 'https://drive.google.com/file/d/1974h-vndIdVvJZEz4S3t4gkmaGFARuok/view?usp=sharing') || exit -1
echo 'download 2012-03-test.tgz'
fi

tar zxvf '2012-03.tgz'
tar zxvf '2012-03/texts/en/fr/en-fr.tgz'
tar zxvf '2012-03/texts/zh/en/zh-en.tgz'
echo 'complete decompression of 2012-03'

cp en-fr/train.tags.en-fr.en iwslt2012_train_en
sed -i '1,6d' iwslt2012_train_en
sed -i '$d' iwslt2012_train_en
cat en-fr/IWSLT12.TALK.dev2010.en-fr.en.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2010_dev_en

cp zh-en/train.tags.zh-en.zh iwslt2012_train_zh
sed -i '1,6d' iwslt2012_train_zh
sed -i '$d' iwslt2012_train_zh
cat zh-en/IWSLT12.TALK.dev2010.zh-en.zh.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2010_dev_zh

rm -r en-fr
rm -r zh-en
echo 'complete train and dev'

tar zxvf '2012-03-test.tgz'
tar zxvf '2012-03-test/texts/en/fr/en-fr.tgz'
tar zxvf '2012-03-test/texts/zh/en/zh-en.tgz'
echo 'complete decompression of 2012-03-test'

cat en-fr/IWSLT12.TED.MT.tst2011.en-fr.en.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2011_test_en
cat en-fr/IWSLT12.TED.MT.tst2012.en-fr.en.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2012_test_en
cat zh-en/IWSLT12.TED.MT.tst2011.zh-en.zh.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2011_test_zh
cat zh-en/IWSLT12.TED.MT.tst2012.zh-en.zh.xml | grep '<seg' |awk '{for(i=3;i<=NF-1;i++){printf $i" "};print ""}' > iwslt2012_test_zh


rm -r en-fr
rm -r zh-en
echo 'complete test'

echo 'finish!'

