#include "Kuhn_Munkras_Algo.h"

/***************************************************************************/
//KM�㷨��������/СȨƥ�����⣩
//�ο����ϣ�
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
        { //(x,y)�������ͼ��
            visy[y] = true;
            if (yMatchX[y] == -1 || findpath(yMatchX[y]))
            {
                yMatchX[y] = x;
                xMatchY[x] = y;
                return true;
            }
        }
        else if (slack[y] > delta)
            slack[y] = delta; //(x,y)���������ͼ����y���ڽ�������
    }
    return false;
}
void KuhnMunkrasAlgo::KM()
{

    for (int x = 0; x < nx; ++x)
    {
        for (int j = 0; j < ny; ++j)
            slack[j] = MAX; //���ﲻҪ���ˣ�ÿ�λ��µ�x��㶼Ҫ��ʼ��slack
        while (true)
        {
            memset(visx, false, sizeof(visx));
            memset(visy, false, sizeof(visy)); //��������ʼ�������������,���ÿ��findpath()��Ҫ����
            if (findpath(x))
                break;
            else
            {
                int delta = MAX;
                for (int j = 0; j < ny; ++j) //��Ϊdfs(x)ʧ��������xһ���ڽ������У�y���ڽ������У��ڶ����
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
                    //�޸Ķ����Ҫ�����е�slackֵ����ȥdelta
                    //������Ϊlx[i] ��С��delta
                    //slack[j] = min(lx[i] + ly[j] -w[i][j]) --j�����ڽ�����--Ҳ��Ҫ����delta���ڶ����
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
    } //���X��������Y��������Ҫ��Y������Ч�㣬�������ѭ��

    for (int i = 0; i < n_x; i++)
        for (int j = 0; j < n_y; j++)
            G[i][j] = - MAX;//δ���ӵĵ�ȨֵΪ0
    for (int i = 0; i<n_x;i++)
        for (int j = 0; j < n_y; j++)
        {
            if (pat[i][j] < thres)
            {
                int data = int(pat[i][j] / finess);
                G[i][j] = -data; //�������СȨֵ����Ȩֵȡ����
            }
        }
}