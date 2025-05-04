#include <iostream>
#include <queue>
#include <stack>
#include <omp.h>
#include<ctime>
#include<chrono>
using namespace std;
using namespace std::chrono;

class graph
{
public:
	int graphAdjacencyMatrix[20][20] = {0}; // in cpp dynamic allocation of array size is not allowed

	void addEdge(int i, int j)
	{
		graphAdjacencyMatrix[i][j] = 1;
		graphAdjacencyMatrix[j][i] = 1;
	}

	void parallelBFS(int startNode, int n)
	{
		queue<int> q;        
		bool visited[20] = {false};

		q.push(startNode);
		visited[startNode] = true;
		cout << startNode << " ";

		while (!q.empty())
		{
			int temp;

#pragma omp critical
			{
				temp = q.front();
				q.pop();
			}

#pragma omp parallel for
			for (int i = 0; i < n; i++)
			{
				cout<< omp_get_thread_num();
				if (graphAdjacencyMatrix[temp][i] == 1 && visited[i] == false)
				{
#pragma omp critical
					{
						visited[i] = true;
						cout << i << " ";
						q.push(i);
					}
				}
			}
		}
		cout << endl;
	}

	void parallelDFS(int startNode, int n)
	{
		stack<int> st;
		bool visited[20] = {false};

		st.push(startNode);
		visited[startNode] = true;

		while (!st.empty())
		{
			int temp;

#pragma omp critical
			{
				temp = st.top();
				st.pop();
				cout << temp << " ";
			}

#pragma omp parallel for
			for (int i = 0; i < n; i++)
			{
				if (graphAdjacencyMatrix[temp][i] == 1 && visited[i] == false)
				{
#pragma omp critical
					{
						visited[i] = true;
						st.push(i);
					}
				}
			}
		}
		cout << endl;
	}
};

int main()
{
	graph g;

	cout << "Enter the number of edges in graph:";
	int edges;
	cin >> edges;

	cout << "Enter the edges in form of (u,v)";
	int u, v;
	for (int i = 0; i < edges; i++)
	{
		cin >> u >> v;
		g.addEdge(u, v);
	}

	cout << "Enter the number of nodes in graph:";
	int nodes;
	cin >> nodes;

	// printing the adjacency matrix
	for (int i = 0; i < nodes; i++)
	{
		for (int j = 0; j < nodes; j++)
		{
			cout << g.graphAdjacencyMatrix[i][j] << " ";
		}
		cout << endl;
	}

	// double BFSstart = omp_get_wtime();
	// g.parallelBFS(0,nodes);
	// double BFSend = omp_get_wtime();
	// cout<<"Time taken by parallel BFS:"<< BFSend-BFSstart<< "seconds."<<endl;

	// double DFSstart = omp_get_wtime();
	// g.parallelDFS(0,nodes);
	// double DFSend = omp_get_wtime();
	// cout<<"Time taken by parallel DFS:"<< DFSend-DFSstart<<"seconds"<<endl;

	auto BFSstart = chrono::high_resolution_clock::now();
	g.parallelBFS(0, nodes);
	auto BFSend = chrono::high_resolution_clock::now();
	chrono::duration<double> bfs_duration = BFSend - BFSstart;
	cout << "Time taken by parallel BFS: " << bfs_duration.count() << " seconds." << endl;

	auto DFSstart = chrono::high_resolution_clock::now();
	g.parallelDFS(0, nodes);
	auto DFSend = chrono::high_resolution_clock::now();
	chrono::duration<double> dfs_duration = DFSend - DFSstart;
	cout << "Time taken by parallel DFS: " << dfs_duration.count() << " seconds." << endl;

	return 0;
}
