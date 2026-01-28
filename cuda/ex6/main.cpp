#include <bits/stdc++.h>
using namespace std;
int partition(vector<int>& a,int low,int high,int len) {
    int privot=a[low];
    int i=low, j=high;
    while(i != j)
    {
        while(i<j && a[j]>=privot)
            j--;
        while(i<j && a[i]<=privot)
            i++;
        if(i<j) // 交换
            swap(a.at(i),a.at(j));
    }
    a[low]=a[i];
    a[i]=privot;
    return i;
}
void GetLeastNumber(vector<int>& input,int n,vector<int>& output,int k)
{
    int start=0,end=n-1;
    int index = partition(input,start,end,n);
    while (index!=k-1)
    {
        if(index>k-1)
        {
            end = index-1;
            index = partition(input,start,end,n);
        }
        else{
            start = index+1;
            index = partition(input,start,end,n);
        }
    }
    for(int i=0;i<k;i++)
        output.push_back(input[i]);
}
int main() {
    vector<int> a = {3,9,2,6,7,1,4,8};
    vector<int> b;
    int k = 4;
    int n = a.size();
    GetLeastNumber(a,n,b,k);
    for(int i=0;i<k;i++)
        cout<<b[i]<<" ";
    return 0;
}