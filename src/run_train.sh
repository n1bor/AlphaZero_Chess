root=/home/owensr/chess

cd $root/data

trainingSets=0
runTimeSecondsNew=3000
runTimeSecondsOld=1800
run=1
while true
do
	if [ -e ${root}/data/newGames ]; then
		mv newGames train
		trainingSets=$((trainingSets+1))
	fi
    if  [ -e ${root}/data/train ]; then
        if  [ ! -e ${root}/data/trainOld ]; then
            mkdir trainOld
        fi
        date >>log.txt
        echo "new data: $run $trainingSets">>log.txt
        echo "new data: $run $trainingSets"
        python3 $root/AlphaZero_Chess/src/train_one.py $run train $runTimeSecondsNew >>train.log
        run=$((run+1))
        for file in `ls train`;do
          mv train/$file trainOld/${trainingSets}_$file
        done
        rmdir train
    else
        if [ -e ${root}/data/trainOld ]; then
            date >>log.txt
            echo "old data: $run $trainingSets">>log.txt
            echo "old data: $run $trainingSets"
            python3 $root/AlphaZero_Chess/src/train_one.py $run trainOld $runTimeSecondsOld >> train.log
            run=$((run+1))
        else
            echo "waiting for 1st dataset"
            sleep 10
        fi
    fi
done
