#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/msg.h>
#include <unistd.h>
#include <time.h>
#include <sys/ipc.h>
void msg_show_atrr(int msg_id,struct msqid_ds msg_info)
{
    int ret = -1;
    sleep(1);
    ret = msgctl(msg_id,IPC_STAT,&msg_info);
    if(-1 == ret)
    {
        printf("获得消息信息失败\n");
        return;
    }
    printf("\n");
    printf("现在队列中的字节数：\nld\n",msg_info.__msg_cbytes);
    printf("队列中消息数：%d\b",(int)msg_info.msg_qnum);
    printf("队列中最大字节数：%d\n",(int)msg_info.msg_qbytes);
    printf("最后发送消息的进程pid:%d\n",msg_info.msg_lspid);
    printf("最后接收消息的进程pid:%d\n",msg_info.msg_lrpid);
    printf("最后发送消息的时间:%s",ctime(&(msg_info.msg_rtime)));
    printf("最后消息变化时间:%s",ctime(&(msg_info.msg_ctime)));
    printf("消息UID是：%d\n",msg_info.msg_perm.uid);
    printf("消息GID是：%d",msg_info.msg_perm.gid);
}
int main() {
    int ret = -1;
    int msg_flags,msg_id;
    key_t key;
    struct msgbuf{
        int mtype;
        char mtext[10];
    };
    struct msqid_ds msg_info;
    struct msgbuf msg_mbuf;
    int msg_sflags,msg_rflags;
    char *msgpath = "/home/liushuai/test/";
    key = ftok(msgpath,'b');
    if(key != -1)
        printf("成功建立Key\n");
    else
        printf("建立KEY失败\n");
    msg_flags = IPC_CREAT|IPC_EXCL;
    msg_id = msgget(key,msg_flags|0x0666);
    if(-1 == msg_id)
    {
        printf("消息建立失败\n");
        return 0;
    }
    msg_show_atrr(msg_id,msg_info);
    msg_sflags = IPC_NOWAIT;
    msg_mbuf.mtype = 10;
    memcpy(msg_mbuf.mtext,"测试消息",sizeof("测试消息"));
    ret = msgsnd(msg_id,&msg_mbuf,sizeof("测试消息"),msg_sflags);
    printf("ret%d\n",ret);
    if(-1 == ret)
        printf("发送消息失败\n");
    else
        printf("接收消息成功，长度:%s\n",ret);
    msg_info.msg_perm.uid = 8;
    msg_info.msg_perm.gid = 8;
    msg_info.msg_qbytes = 12345;
    ret = msgctl(msg_id,IPC_SET,&msg_info);
    if(-1 == ret)
    {
        printf("设置消息属性失败\n");
        return 0;
    }
    msg_show_atrr(msg_id,msg_info);
    ret = msgctl(msg_id,IPC_RMID,nullptr);
    if(-1 == ret)
    {
        printf("删除消息失败\n");
        return 0;
    }
    return 0;
}