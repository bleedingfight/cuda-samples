#include <bits/stdc++.h>
using namespace std;
class Node {
  public:
    int data;
    Node *next;
    Node(int d, Node *p) {
        data = d;
        next = p;
    }
};
Node *FindKthToTail(Node *head, unsigned int k) {
    // k=0或者指针为空时可能导致崩溃
    if (head == nullptr || k == 0)
        return nullptr;
    int i = 0;
    Node *p = head;
    Node *q = head;
    Node *finded_node = new Node(0, nullptr);
    for (int j = 0; j < k; j++) {
        // 每一步都必须判断是不是走到了空指针位置
        if (p != nullptr)
            p = p->next;
        else
            return nullptr;
    }
    while (p) {
        p = p->next;
        q = q->next;
    }

    finded_node->data = q->data;
    finded_node->next = q->next;
    return finded_node;
}
Node *find_k(Node *phead, int k) {
    int count = 0, i = 0;
    Node *find = phead;
    if (phead == NULL)
        return NULL;
    while (find->next != NULL) {
        count++;
        find = find->next;
    }
    if (k > count) {
        printf("K is too big. \n");
        exit(1);
    }
    find = phead;
    while (i < count - k + 1) {
        find = find->next;
        i++;
    }
    return find;
}
Node *string_to_linklist(string &a) {
    queue<int> num_queue;
    for (auto i : a) {
        int num = i - '0';
        if (num >= 0 && num <= 9)
            num_queue.push(i);
    }
    Node head(0);
    if (num_queue.size() > 0) {
        Node head = Node(num_queue.front());
        num_queue.pop();
    }
    while (num_queue.size() > 0) {
        Node current = Node(num_queue.front());
        head num_queue.pop();
        if
    }
}
void show_stack(stack<int> s) {
    while (s.size() > 0) {
        cout << s.top() << " ";
        s.pop();
    }
}
void print_rkth(string &str_input, int k) {
    stack<int> s;
    for (auto i : str_input) {
        int num = i - '0';
        if (num >= 0 && num <= 9)
            s.push(num);
    }
    if (k <= 0 || k >= s.size())
        return;
    else {
        if (k == 1)
            cout << s.top() << "\n";
        else {
            for (int i = 1; i < k; i++)
                s.pop();
            cout << s.top() << "\n";
        }
    }
}
int main() {
    Node *root = new Node(1, nullptr);
    Node *a = new Node(1, nullptr);
    Node *b = new Node(2, nullptr);
    Node *c = new Node(8, nullptr);
    //    string str_input = "1->1->2->8";
    stack<int> s;
    string str_input;
    int rkth;
    cin >> str_input >> rkth;
    //    int rkth = c
    print_rkth(str_input, rkth);

    root->next = a;
    a->next = b;
    b->next = c;

    Node *find_node;
    //    find_node = FindKthToTail(root,2);
    find_node = find_k(root, 2);
    //    cout<<find_node->data<<" ";
    delete root;
    delete a;
    delete b;
    delete c;
    delete find_node;

    return 0;
}
