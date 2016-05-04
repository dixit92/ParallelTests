// Name: Smit, Sehal Dixit and Praharshith Gurajala
// Large Project Part 2 (MPI/OMP Hybrid program)
//Run OMP/Shared memory on a single node and run MPI between nodes
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <queue>
#include <map>
#include <iterator>
#include <omp.h>
#include <stdint.h>
#include <string.h>
#include "mpi.h"

//This is PER NODE
//1 read thread per node only - this is because it acts as a recieve threads
//and does memory management, not disk io
#define NUM_READ_THREADS   1
#define NUM_MAP_THREADS    4
#define NUM_REDUCE_THREADS 4

// List of files to apply MapReduce on
char * all_files_list[] = {"f1.txt", "f2.txt", "f3.txt", "f4.txt", "f5.txt", "f6.txt", "f7.txt", "f8.txt"};
#define NFILES 8

// Hashing function that will be used to assign work to reducer threads
uint32_t SuperFastHash (const char *, int);

// Used in SuperFastHash function
#define get16bits(d) (*((const uint16_t *) (d)))

//Global parameters
int mpirank, nodes;
char *result_text;
int resultlen;

using namespace std;

//declare OMP portion of the program
void do_omp_mr (char *, int);

//Serialize map
void serializemap (map<string,int>, char *, int);

//Deserialize map
map<string,int> deserializemap(char *, int);

int main (int argc, char *argv[])
{
	//Initialize MPI on 1 process per node
	int provided;
//	MPI_Init_thread (NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
	MPI_Init(&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &mpirank);
	MPI_Comm_size (MPI_COMM_WORLD, &nodes);

	//The first node will scatter files equally to all nodes
	//First, all files must be read into a buffer by the root process
	//Timing the disk I/O operation

	double iostart, iostop, iotime;

	iostart = MPI_Wtime();

	string input_text [NFILES];				//All the text
	int *nodelen;											//Array holding no of characters for all nodes
	char *node_text;									//Text on one node
	int *textlen;											//Number of characters on that node

	textlen = new int;

	if (mpirank == 0)
	{
		cout <<endl<< "Nodes: " <<nodes<<endl;
		cout <<"Files: "<<NFILES<<endl
		<<"Map Threads per node: "<<NUM_MAP_THREADS<<endl
		<<"Reduce Threads per node: "<<NUM_REDUCE_THREADS<<endl
		<<"Starting MapReduce..."<<endl<<endl;
		for (int i=0; i<NFILES; i++)
		{
			ifstream in(all_files_list[i]);
		  stringstream filebuffer;
			filebuffer << in.rdbuf();
		  input_text[i] = filebuffer.str();
		}


	}

	iostop = MPI_Wtime();
	iotime = iostop - iostart;
//	cout <<"Print 1"<<endl;
	//This barrier ensures that the root process has completed reading the file
	MPI_Barrier(MPI_COMM_WORLD);

	//Time measurement for initial collective communication
	double commstart, commstop, commtime;
	commstart = MPI_Wtime();


	//Scatter the contents of the files among all nodes
	int filespernode = NFILES/nodes;

	//First, we need to send information about the length of the files to each nodes
	//To do this, we need to calculate total number of characters to be sent to each
	//node from at the master node
	if (mpirank == 0)
	{
		nodelen = new int[nodes];

		//Zero the nodelen array
		int i;
		for (i=0; i<nodes; i++)
		{
			nodelen[i] = 0;
		}

		int activenode = 0;
		//Chars per node
		for (i=0; i<NFILES; i++)
		{
			if ( (i%filespernode==0) && i!=0)
				activenode++;
			nodelen[activenode] = nodelen[activenode] + input_text[i].length();
		}
	}

	MPI_Scatter (nodelen, 1, MPI_INT, textlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
//	cout <<"Print 2"<<endl;
	//Allocate memory on nodes to recieve the Chars
	node_text = new char[*textlen];

	MPI_Status status;
	MPI_Request request;

	//Now to send the actual data
	//for the root process
	if (mpirank == 0)
	{
		int i, j;

		string temp;

		//First, initialize the text to be processed on the root node
		for (i=0; i<filespernode; i++)
			temp = temp + input_text[i];

		strcpy (node_text, temp.c_str());

		//Next, send the text to be processed to all other nodes
		if (nodes > 1)
		{
			for (i=1; i<nodes; i++)
			{
				temp = "";

				for (j=i*filespernode; j<i*filespernode + filespernode; j++)
					temp = temp + input_text[j];

				char *tosend;
				tosend = new char[nodelen[i]];
				strcpy (tosend, temp.c_str());

				MPI_Send (tosend, nodelen[i], MPI_CHAR, i, 0, MPI_COMM_WORLD);
			}
		}
	}

	if (mpirank != 0)
	{
		MPI_Recv (node_text, *textlen, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
	}
	//cout <<"Print 3"<<endl;
	//Debuging
	/*
	string outfile;
	stringstream ss;
	ss << mpirank << ".txt";
	outfile = ss.str();
	ofstream out(outfile.c_str());
	out << node_text;
*/

	//End of file i/o and distribution to all processess
	//Timing barrier
	MPI_Barrier (MPI_COMM_WORLD);
	commstop = MPI_Wtime();
	commtime = commstop - commstart;

	//Begin MapReduce
	double mrstart, mrstop, mrtime;

	mrstart = MPI_Wtime();

	do_omp_mr (node_text, *textlen);
	//cout <<"Print 4"<<endl;
	mrstop = MPI_Wtime();
	mrtime = mrstop - mrstart;

	//End MapReduce
	MPI_Barrier (MPI_COMM_WORLD);

	//Collect all wordcounts
	commstart=MPI_Wtime();

	string allresults;						//The collected and aggregated reducer results
	int *allresultslen;					//Array containing lengths of results from each node

	if (mpirank == 0)
	{
		allresultslen = new int[nodes];
	}

	//Collect all lengths of results
	MPI_Gather (&resultlen, 1, MPI_INT, allresultslen, 1, MPI_INT, 0, MPI_COMM_WORLD);
//	cout <<"Print 5"<<endl;
	//Now that we have the result lengths get the resutls from all processess
	//and aggregated
	if (mpirank == 0)
	{
		int i, j;

		allresults = "";

		allresults = allresults + result_text;

		char *temp;

		if (nodes > 1)
		{
			for (i=1; i<nodes; i++)
			{
				temp = new char[allresultslen[i]];
				MPI_Recv(temp, allresultslen[i], MPI_CHAR, i, 1, MPI_COMM_WORLD, &status);
				allresults.append(temp, allresultslen[i]);
				delete temp;
			}
		}
	}

	if (mpirank!=0)
	{
		MPI_Send (result_text, resultlen, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
	}
	//cout <<"Print 6"<<endl;
	MPI_Barrier (MPI_COMM_WORLD);
	commstop=MPI_Wtime();
	commtime += commstop - commstart;

	//Collected all wordcounts, now output to file in root process
	iostart = MPI_Wtime();

	if (mpirank == 0)
	{
		ofstream finalout("output.txt");
		finalout << allresults;
		finalout.close();
	}

	iostop = MPI_Wtime();
	iotime += iostop - iostart;
	//All done, print summary
	if (mpirank == 0)
	{
		cout<<endl<<"...complete! See output.txt";
		cout<<endl<<"I/O time total (files on disk): "<<iotime;
		cout<<endl<<"I/O network Communications time: "<<commtime;
		cout<<endl<<"MapReduce Execution time on 1 node: "<<mrtime;
		cout<<endl<<"Total time taken: "<<iotime+commtime+mrtime<<endl;
	}

	MPI_Finalize();
}

void do_omp_mr(char * text, int length) {
  int i, j, k, nthreads;
  int num_files_done = 0;

	double maptimestart, maptimestop, maptime;
	double reducetimestart, reducetimestop, reducetime;
	double ncommtimestart, ncommtimestop, ncommtime;

  queue<string>     			read_q;
  map<string,int>   			*mapwords;
  mapwords = new map<string,int> [NUM_MAP_THREADS];

  queue < pair<string,int> > 	*reduce_q;
  reduce_q = new  queue < pair<string,int> > [nodes*NUM_REDUCE_THREADS];
  map<string,int> 		*reducewords;
  reducewords = new map<string,int> [nodes*NUM_REDUCE_THREADS];

	map<string,int>			*finalreduce;
	finalreduce = new map<string, int> [nodes*NUM_REDUCE_THREADS];

  omp_lock_t *reduce_q_lock;
  reduce_q_lock = new omp_lock_t[nodes*NUM_REDUCE_THREADS];

  for (i=0; i<NUM_REDUCE_THREADS * nodes; i++)
    omp_init_lock(&(reduce_q_lock[i]));
//	cout <<"Print 7"<<endl;
  // Run Mapper threads on recieved file

	maptimestart = MPI_Wtime();
  #pragma omp parallel num_threads(NUM_READ_THREADS + NUM_MAP_THREADS) shared(reducewords, reduce_q_lock)
  {
    int tid = omp_get_thread_num();

    // All threads with tid < NUM_READ_THREADS are considered Reader threads
    if (tid < NUM_READ_THREADS) {

			stringstream textstr(text);
			string word;

      // Read the txtstr and split on newline. Assign each element to word
        // Since the read_q is shared among reader threads and mapper threads,
        // only one reader thread should push words into the queue at a time
		 if (textstr != NULL)
		 {
			while (getline(textstr, word, '\n'))
			{
			#pragma omp critical
        {
          read_q.push(word);
        }
			}
		 }
			num_files_done++;

//	cout<<"End of read on node "<<mpirank<<" thread "<<tid<<endl;
    } // End of Reader threads
//    The above block has been tested to work



    // All threads with tid >= NUM_READ_THREADS are considered Mapper threads
    else if(tid >= NUM_READ_THREADS) {
    int id = tid - NUM_READ_THREADS;
    string mapword = "";
    bool empty = false;
    map<string,int>::iterator 				iter;
    map<string,int >::const_iterator 	i_iter;

  	while(num_files_done < NUM_READ_THREADS || empty == false) {
			#pragma omp critical
        {
          empty = read_q.empty();
          if(empty == false) {
            mapword = read_q.front();
            read_q.pop();
          }
          else
            mapword="";
        }

        if (mapword != "") {
          iter = mapwords[id].find(mapword);
          if(iter == mapwords[id].end()) {
            mapwords[id].insert(map<string, int>::value_type(mapword, 1));
          }
          else {
            iter->second = iter->second + 1;
          }
        }
      } // End of while loop

  //    cout<<"End of map stage 1 on node "<<mpirank<<" thread "<<tid<<endl;
	//The above section has been verified correct

      // Mapper threads have finished calculating the word counts.
      // Now each Mapper thread will now send their words to reducer thread
      // based on the hash value
			//The hashrange is determined by the MPI rank of the node and the number
			//of nodes
      int    hash;
      string hashword;
      int    hashcount;

      for(i_iter = mapwords[id].begin(); i_iter != mapwords[id].end(); i_iter++) {
          hashword = 	i_iter->first;
          hashcount = i_iter->second;
          hash = SuperFastHash(hashword.c_str(), hashword.length());

      //cout<<"For iter"<<i_iter->first<< " on node "<<mpirank<<" thread "<<tid<<endl;

          // Push each word, count pair to a reduce queue based on the hash value
          omp_set_lock(&(reduce_q_lock[hash]));
          //reducewords[hash].push(make_pair(hashword, hashcount));
          iter = reducewords[hash].find(hashword);
          if(iter == reducewords[hash].end()) {
            reducewords[hash].insert(map<string, int>::value_type(hashword, hashcount));
          }
          else {
            iter->second = iter->second + hashcount;
          }
          omp_unset_lock(&(reduce_q_lock[hash]));
      }

  //    cout<<"End of map stage 2 on node "<<mpirank<<" thread "<<tid<<endl;
    }

    //Above code has been checked to be correct
		maptimestop = MPI_Wtime();
		maptime = maptimestop - maptimestart;

     // End of Mapper threads
  } // End of parallel section
//  cout <<"Print 9"<<endl;
  for (int i=0; i<NUM_REDUCE_THREADS * nodes; i++)
    omp_destroy_lock(&(reduce_q_lock[i]));

	//MPI Communications between reducers begins
	//32-bit unsigned int has a value between 0 and 4294967295
	//Reducewords is split into nodes * NUM_REDUCE_THREADS parts according
	//To hash
	MPI_Status status;
	//Scaffold MPI Code
	//Each process must send the words in the range of the respective hash
	//To other processes

	//First, serialize the reducewords maps

	int * seriallen;
	int * recievedlen;
	char ** serializedmap;
	char ** recievedmap;

	seriallen = new int[nodes*NUM_REDUCE_THREADS];
	serializedmap = new char*[nodes*NUM_REDUCE_THREADS];
	recievedlen = new int[nodes*NUM_REDUCE_THREADS];
	recievedmap = new char*[nodes*NUM_REDUCE_THREADS];



	cout<<"Comm memory allocated on node "<<mpirank;

	//#pragma omp parallel for() //This can be done in omp
	for (int i=0; i<nodes*NUM_REDUCE_THREADS; i++)
	{
		//serializemap (reducewords[i], serializedmap[i], seriallen[i]);

		string serialmap = "";
		stringstream ss;
		map<string, int>::iterator iter;
		for (iter = reducewords[i].begin(); iter != reducewords[i].end(); iter++)
		{
			ss << iter->first << " " << iter->second << "\n";
		}

		serialmap = ss.str();
		seriallen[i] = serialmap.length();
		serializedmap[i] = new char[seriallen[i]];
		strcpy (serializedmap[i], serialmap.c_str());
	}
	cout <<"Print 10: Serialization complete on node "<<mpirank<<endl;
	//Next, send and recieve the length information
	//All nodes must call this to scatter respective Mapped elements to their
	//reduce nodes

	MPI_Barrier(MPI_COMM_WORLD);
	ncommtimestart = MPI_Wtime();

	//Print lengths before sending
	for (int i=0; i<NUM_REDUCE_THREADS*nodes; i++)
		if (mpirank == 1)
			cout <<"Len on  "<<i<<" on node "<<mpirank<<" is "<<seriallen[i]<<endl;

	MPI_Barrier(MPI_COMM_WORLD);
	for (int i=0; i<NUM_REDUCE_THREADS*nodes; i++)
		if (mpirank == 0)
			cout <<"Len on  "<<i<<" on node "<<mpirank<<" is "<<seriallen[i]<<endl;

	MPI_Barrier(MPI_COMM_WORLD);



	MPI_Status *mapstat, *nosstat;
	nosstat = new MPI_Status[nodes * NUM_REDUCE_THREADS];
	mapstat = new MPI_Status[nodes * NUM_REDUCE_THREADS];
	MPI_Request *mapreq;
	mapreq = new MPI_Request[nodes * NUM_REDUCE_THREADS];


	int ic1, ic2, ic;

	for (int i=0; i<nodes; i++)
	{
		if (i!=mpirank)
		{
			ic2 = i*NUM_REDUCE_THREADS;

			for (int j=0; j<NUM_REDUCE_THREADS; j++)
			{
				MPI_Send(&seriallen[ic2+j], 1, MPI_INT, i, j, MPI_COMM_WORLD);
			}
		}
	}

	for (int i=0; i<nodes; i++)
	{
		if (i==mpirank)
		{
			for (int k = i*NUM_REDUCE_THREADS;k < i*NUM_REDUCE_THREADS + NUM_REDUCE_THREADS; k++)
				recievedlen[k] = seriallen[k];
		}
		else
		{
			ic1 = i*NUM_REDUCE_THREADS;

			for (int j=0; j<NUM_REDUCE_THREADS; j++)
			{
				MPI_Recv(&recievedlen[ic1+j], 1, MPI_INT, i, j, MPI_COMM_WORLD, &nosstat[ic1+j]);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//Print recieved lengths
	for (int i=0; i<NUM_REDUCE_THREADS*nodes; i++)
		if (mpirank == 0)
			cout <<"Recieved map len"<<i<<" on node "<<mpirank<<" is "<<recievedlen[i]<<endl;

	MPI_Barrier(MPI_COMM_WORLD);

	for (int i=0; i<NUM_REDUCE_THREADS*nodes; i++)
		if (mpirank == 1)
			cout <<"Recieved map len"<<i<<" on node "<<mpirank<<" is "<<recievedlen[i]<<endl;

	MPI_Barrier(MPI_COMM_WORLD);


	cout <<"Print 11"<<" length info recieved on node" <<mpirank<<endl;
	//Allocate memory for incoming maps
	for (int i=0; i<NUM_REDUCE_THREADS*nodes; i++)
	{
		recievedmap[i] = new char[recievedlen[i]+1];
	}

	MPI_Barrier(MPI_COMM_WORLD);


	//Allocate memory for aggregated len

	cout <<"Print 21 Memory allocation complete on "<<mpirank<<endl;
	//Now, exchange the serialized Map



	MPI_Barrier(MPI_COMM_WORLD);
	
	i = mpirank;
	for (int k = i*NUM_REDUCE_THREADS;k < i*NUM_REDUCE_THREADS + NUM_REDUCE_THREADS; k++)
		{
				strcpy(recievedmap[k], serializedmap[k]);
				cout <<"Node "<<i<<" copied map "<<k<<endl;
		}	
	for (int i=0; i<nodes; i++)
	{
		if (i!=mpirank)

		{
			ic2 = i*NUM_REDUCE_THREADS;

			for (int j=0; j<NUM_REDUCE_THREADS; j++)
			{
				MPI_Irecv(recievedmap[ic2+j], recievedlen[ic2+j], MPI_CHAR, i, 0, MPI_COMM_WORLD, &mapreq[ic2+j]);

				cout <<"Print 16.5 recieved map"<<ic2+j<<" from node "<<mpirank<<endl;

			}
		}
	}

	for (int i=0; i<nodes; i++)
	{
		if (i!=mpirank)
		{
			ic1 = i*NUM_REDUCE_THREADS;

			for (int j=0; j<NUM_REDUCE_THREADS; j++)
			{
			
			
				cout <<"Print 16 sent map"<<ic1+j<<" from node "<<mpirank
				<<" to node "<<i<<" len "<<seriallen[ic1+j]<<endl;
				MPI_Ssend(serializedmap[ic1+j], seriallen[ic1+j], MPI_CHAR, i, 0, MPI_COMM_WORLD);

//
			}
		}
	}
	

	MPI_Barrier(MPI_COMM_WORLD);
	ncommtimestop = MPI_Wtime();
	ncommtime = ncommtimestop - ncommtimestart;
	cout <<"Print 12 recieved all serialized maps on node "<<mpirank<<endl;

	stringstream aggregatedmap[NUM_REDUCE_THREADS];

	//Concatenate data for each reducer recieved from the mappers on all nodes
	for (int i=0; i<NUM_REDUCE_THREADS*nodes; i++)
	{
		recievedmap[i][recievedlen[i]] = '\0';

		aggregatedmap[i%NUM_REDUCE_THREADS] << recievedmap[i];
//		cout<<"Concat for aggregated "<<i%NUM_REDUCE_THREADS<<" recieved "<<i
	//	<<" node "<<mpirank<<endl;
	}


//	cout<<"Aggregation complete on node"<<mpirank<<" len"<<aggregatedmap[i].str().length()
//	<<" actual "<<aggregatedmap[0].str()<<endl;
	MPI_Barrier(MPI_COMM_WORLD);


	//Now deserialize
	//map<string, int> reducewords[NUM_REDUCE_THREADS];

	for (int i=0; i<NUM_REDUCE_THREADS; i++)
	{
		//reducewords[i] = deserializemap(aggregatedmap[i], aggregatedlen[i]);
		string ss;
		string word;
		int count;

		while (getline(aggregatedmap[i], ss))
		{

				int space = ss.find(" ");
				word = ss.substr(0, space-1);

				stringstream convert (ss.substr(space, ss.length()));
				convert >> count;
				finalreduce->insert(pair<string, int>(word, count));
		}

	//	cout<<"Deserialized aggregatedmap "<<i<<" on node "<<mpirank<<endl;
	}
//	cout <<"Print 13, Deserialization complete on node "<<mpirank<<endl;
  // Specify the output stringstream to write resutls is resultstr
	stringstream resultstr;

// Reducer threads to combine all word count pairs from each mapper

reducetimestart = MPI_Wtime();
//NOTE: YOU NEED TO CHANGE REDUCEWORDS MAP TO THE ONE RECIEVED FROM MPI
	#pragma omp parallel num_threads(NUM_REDUCE_THREADS)
  {
    int id = omp_get_thread_num();
   // pair<string,int> reducewords;
   // reducewords = recvmap;
    map <string,int>::iterator iter;
    map <string,int >::const_iterator i_iter;

    // Write into the output file
    //only one reducer thread should work on one word at a time
    #pragma omp critical
    for(i_iter = finalreduce[id].begin(); i_iter != finalreduce[id].end(); i_iter++) {
      resultstr << i_iter->first << " -> " << i_iter->second << "\n";
    }
  } // End of parallel section
	reducetimestop = MPI_Wtime();
	reducetime = reducetimestop - reducetimestart;

//  cout <<"Print 14 reducetion complete on node "<<mpirank<<endl;

	MPI_Barrier(MPI_COMM_WORLD);

	cout<<"Time taken for Map on node "<<mpirank<<maptime<<endl;
	cout<<"Time taken for reduce on node "<<mpirank<<": "<<reducetime<<endl;
	cout<<"Time taken for internode comms on node "<<mpirank<<": "<<ncommtime<<endl;

  //Converting resultstr to str (updated 6pm, 02-05):
  resultlen = resultstr.str().length();
  result_text = new char[length];
  strcpy (result_text, resultstr.str().c_str());

}

uint32_t SuperFastHash (const char * data, int len) {
  uint32_t hash = len, tmp;
  int rem;

    if (len <= 0 || data == NULL) return 0;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
                hash ^= hash << 16;
                hash ^= ((signed char)data[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (data);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*data;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash % (nodes*NUM_REDUCE_THREADS);
}

void serializemap (map<string,int> mymap, char *res, int &len)
{
	string serialmap = "";
	stringstream ss;
	int i;
	map<string, int>::iterator iter;
	for (iter = mymap.begin(); iter != mymap.end(); iter++)
	{
		ss << iter->first << " " << iter->second << "\n";
	}

	serialmap = ss.str();
	len = serialmap.length();
	res = new char[len];
	strcpy (res, serialmap.c_str());
}

map<string, int> deserializemap (char *inputchar, int charlen)
{
	map<string, int> outmap;

	inputchar[charlen-1] = '\0';

	stringstream str(inputchar);
	string ss;
	string word;
	int count;

	while (getline(str, ss))
	{

			int space = ss.find(" ");
			word = ss.substr(0, space-1);

			stringstream convert (ss.substr(space, ss.length()));
			convert >> count;
			outmap.insert(pair<string, int>(word, count));
	}

	return outmap;
}
