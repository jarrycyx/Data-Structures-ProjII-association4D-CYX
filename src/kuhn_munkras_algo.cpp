#include "Kuhn_Munkras_Algo.h"

/***************************************************************************/
//KM算法（解决最大/小权匹配问题）
//参考资料：
//https://blog.csdn.net/sixdaycoder/article/details/47720471
//https://www.cnblogs.com/zpfbuaa/p/7218607.html
/***************************************************************************/

bool KuhnMunkrasAlgo::findpath(int x)
{
    int delta;

    visx[x] = true;
    for (int y = 0; y < ny; ++y)
    {
        if (visy[y])
            continue;
        delta = lx[x] + ly[y] - G[x][y];
        if (delta == 0)
        { //(x,y)在相等子图中
            visy[y] = true;
            if (yMatchX[y] == -1 || findpath(yMatchX[y]))
            {
                yMatchX[y] = x;
                xMatchY[x] = y;
                return true;
            }
        }
        else if (slack[y] > delta)
            slack[y] = delta; //(x,y)不在相等子图中且y不在交错树中
    }
    return false;
}
void KuhnMunkrasAlgo::KM()
{

    for (int x = 0; x < nx; ++x)
    {
        for (int j = 0; j < ny; ++j)
            slack[j] = MAX; //这里不要忘了，每次换新的x结点都要初始化slack
        while (true)
        {
            memset(visx, false, sizeof(visx));
            memset(visy, false, sizeof(visy)); //这两个初始化必须放在这里,因此每次findpath()都要更新
            if (findpath(x))
                break;
            else
            {
                int delta = MAX;
                for (int j = 0; j < ny; ++j) //因为dfs(x)失败了所以x一定在交错树中，y不在交错树中，第二类边
                    if (!visy[j] && delta > slack[j])
                        delta = slack[j];
                for (int i = 0; i < nx; ++i)
                    if (visx[i])
                        lx[i] -= delta;
                for (int j = 0; j < ny; ++j)
                {
                    if (visy[j])
                        ly[j] += delta;
                    else
                        slack[j] -= delta;
                    //修改顶标后，要把所有的slack值都减去delta
                    //这是因为lx[i] 减小了delta
                    //slack[j] = min(lx[i] + ly[j] -w[i][j]) --j不属于交错树--也需要减少delta，第二类边
                }
            }
        }
    }
}
int KuhnMunkrasAlgo::solve(int* yToX, int* xToY)
{
    memset(yToX, -1, sizeof(yMatchX));
    memset(xToY, -1, sizeof(xMatchY));
    memset(yMatchX, -1, sizeof(yMatchX));
    memset(xMatchY, -1, sizeof(xMatchY));
    memset(ly, 0, sizeof(ly));
    for (int i = 0; i < nx; ++i)
    {
        lx[i] = -MAX;
        for (int j = 0; j < ny; ++j)
            if (lx[i] < G[i][j])
                lx[i] = G[i][j];
    }
    KM();

    int ans = 0;
    for (int i = 0; i < ny; ++i)
    {
        if (G[yMatchX[i]][i] > (-MAX / 2)) {
            yToX[i] = yMatchX[i];
            if (yMatchX[i] != -1)
                ans += G[yMatchX[i]][i];
        }
    }
    for (int i = 0; i < nx; ++i)
        if (G[i][xMatchY[i]] > (-MAX / 2))
            xToY[i] = xMatchY[i];

    return ans;
}


void KuhnMunkrasAlgo::loadData(int n_x, int n_y, double pat[][MAXN], double finess, double thres)
{
    if (n_x > n_y)
    {
        nx = n_x;
        ny = n_x;
    }
    else 
    {
        nx = n_x;
        ny = n_y;
    } //如果X数量大于Y数量，需要在Y补上无效点，否则会死循环

    for (int i = 0; i < n_x; i++)
        for (int j = 0; j < n_y; j++)
            G[i][j] = - MAX;//未连接的点权值为0
    for (int i = 0; i<n_x;i++)
        for (int j = 0; j < n_y; j++)
        {
            if (pat[i][j] < thres)
            {
                int data = int(pat[i][j] / finess);
                G[i][j] = -data; //求的是最小权值，故权值取负数
            }
        }
}