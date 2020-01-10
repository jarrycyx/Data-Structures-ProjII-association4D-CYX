#pragma once

#include <cstring>
#include <cstdio>
#include <iostream>

#define MAXN 120
#define MAX 0x3f3f3f3f

using namespace std;

/***************************************************************************/
//KM算法（解决最大权匹配问题）
//参考资料：
//https://blog.csdn.net/sixdaycoder/article/details/47720471
//https://www.cnblogs.com/zpfbuaa/p/7218607.html
/***************************************************************************/

class KuhnMunkrasAlgo
{
public:
	void KM();
	int solve(int* yToX, int* xToY);
	bool findpath(int x);
	void loadData(int n_x, int n_y, double pat[][MAXN], double finess, double thres);
private:
	int yMatchX[MAXN], xMatchY[MAXN], lx[MAXN], ly[MAXN], slack[MAXN];
	int G[MAXN][MAXN];
	bool visx[MAXN], visy[MAXN];
	int n, nx, ny;
};

