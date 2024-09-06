root=/home/owensr/chess
runTimeSeconds=3600
cd ${root}/data
mkdir games
while true
do
	
	for i in `seq 2`
	do
	python3 $root/AlphaZero_Chess/src/generate.py $i $runTimeSeconds &
	sleep 1
	done
	wait
	while [ -e newGames ]; do
		echo waiting 10 secs for training to finish
		sleep 10
	done
	mv games newGames
	mkdir games

done
