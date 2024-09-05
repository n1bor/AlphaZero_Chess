root=/home/owensr/chess
runTimeSeconds=3600
cd ${root}/data
mkdir games
while true
do
	if [ -e newGames ]; then
		echo waiting 60 secs for training to finish
		sleep 10
	else
		for i in `seq 2`
		do
		python3 $root/AlphaZero_Chess/src/generate.py $i $runTimeSeconds &
		sleep 1
		done
		wait
		mv games newGames
		mkdir games
	fi
done
