#include<iostream>
using namespace std;
#include<string>
#include<vector>
#include<queue>
#include<unordered_map>
#include<algorithm>
#include<numeric>
#include<stack>
#include<time.h>
#include<iterator>
#include<unordered_set>
#include<functional>
#include<sstream>
#include<cstring>
#include<thread>
#include<mutex>
#include<memory>
#define NDEBUG
#include<assert.h>

//浮点数比较（数值分析之相对误差和绝对误差）
const float RELATIVE_ERROR = 1e-6;
const float ABSOLUTE_ERROR = 1e-6;
bool IsEqual(float a, float b, float absError)//, float relError
{
	if (a == b) return true;
	return fabs(a - b) < absError;
	/*cout << fabs(a - b)/fabs(a) << endl;
	if (a>b) return  (fabs((a - b) / a)>relError) ? true : false;
	cout << "line 30"<< endl;
	return  (fabs((a - b) / b)>relError) ? true : false;*/
}
//cpp:split()
void split(const string& s, vector<string>& vret, const char flag = ' ') {
	vret.clear();
	istringstream ss(s);
	string tmp;
	while (getline(ss, tmp, flag))
		vret.emplace_back(tmp);
	return;
}
//单例模式
std::mutex mtx;
int index = 0;
void printArray(const vector<int>& v)
{

	while (index<v.size())
	{
		if (mtx.try_lock()) {
			std::cout << v[index++] << "\t";
			mtx.unlock();
		}

	}
}
class Singleton {//懒汉单例模式
private:
	Singleton() { cout << "getInstance" << endl; }
	Singleton(const Singleton& obj) {}
	Singleton& operator=(const Singleton& obj) {}
	static Singleton* instance;
	static mutex mtx;
public:
	~Singleton() { delete instance; }
	static Singleton* getInstance();
};
mutex Singleton::mtx;
Singleton* Singleton::instance = nullptr;
Singleton* Singleton::getInstance() {
	if (instance == nullptr)
		if (mtx.try_lock()) {
			if (instance == nullptr)
				instance = new Singleton;
			mtx.unlock();
		}
	return instance;
};
class SingletonForHungry {//饿汉单例模式，多线程访问无需手动同步，它是自动同步的(getinstance()每次只是返回一个地址而已，并不用创建新对象，所以线程访问无需加锁)
private:
	SingletonForHungry() { cout << "getInstance" << endl; }
	SingletonForHungry(const SingletonForHungry& obj) {}
	SingletonForHungry& operator= (const SingletonForHungry& obj) {}
	static SingletonForHungry* instance;
public:
	~SingletonForHungry() { delete instance; }
	static SingletonForHungry* getInstance();
};
SingletonForHungry* SingletonForHungry::instance = new SingletonForHungry;
SingletonForHungry* SingletonForHungry::getInstance() {
	return instance;
}
//编解码
string decodeString(const string& s) {
	int n = s.size();
	string sret;
	int i = 0;
	while (i < n) {
		int num = 0;
		while (s[i] >= '0'&&s[i] <= '9')
			num = 10 * num + s[i++] - '0';
		if (s[i] == '[') ++i;
		string  ss;
		while (s[i] >= 'a'&&s[i] <= 'z')
			ss += s[i++];
		if (s[i] == ']') ++i;
		if (num == 0) num = 1;
		for (int j = 0; j < num; ++j)
			sret += ss;

	}
	return sret;
}
string inplace(string& s, string& ss, string::iterator i, string::iterator j) {
	string sret;
	copy(s.begin(), i, back_inserter(sret));
	sret += ss;
	copy(++j, s.end(), back_inserter(sret));
	return sret;
}
string decodeStr(string& s) {
	while (find(s.begin(), s.end(), ']') != s.end()) {
		auto i = s.begin();
		while (*i != ']') ++i;
		auto j = i;
		while (*j != '[') --j;
		if (*j == '[') --j;
		while (j > s.begin() && *j >= '0'&& *j <= '9') --j;
		string ss;
		if (j == s.begin()) ss = decodeString(string(j, i));
		else ss = decodeString(string(++j, i));
		s = inplace(s, ss, j, i);
	}
	return s;
}

struct TreePNode {
	int value;
	TreePNode* left = nullptr;
	TreePNode* right = nullptr;
	TreePNode* parent = nullptr;
	TreePNode(int val) :value(val) {}
};
TreePNode* getNext(TreePNode* node) {
	if (node == nullptr) return node;
	TreePNode* ret = nullptr;
	TreePNode* cur = nullptr;
	if (node->right) {
		cur = node->right;
		while (cur->left) cur = cur->left;
		ret = cur;
	}
	else {
		cur = node;
		TreePNode* parent = cur->parent;
		while (parent&&parent->right == cur) {
			cur = parent;
			parent = cur->parent;
		}
		ret = parent;
	}
	return ret;
}
struct TreeNode {
	int value;
	TreeNode* left = nullptr;
	TreeNode* right = nullptr;
	TreeNode(int val) :value(val) {}
};
class ConvertBstToList {
public:
	TreeNode* BSTtoDoubleList(TreeNode* root) {
		if (root == nullptr) return root;
		TreeNode* p = nullptr;
		convertTreeToList(root, &p);
		while (p&&p->left)
			p = p->left;
		return p;
	}
private:
	void convertTreeToList(TreeNode* root, TreeNode** prev) {
		if (root == nullptr) return;
		TreeNode* cur = root;

		if (cur->left)
			convertTreeToList(root->left, prev);
		cur->left = *prev;
		if (*prev) (*prev)->right = cur;
		*prev = cur;
		if (cur->right)
			convertTreeToList(root->right, prev);
	}
};

TreeNode* buildTree(TreeNode* root, int val) {
	TreeNode* cur = new TreeNode(val);
	if (root == nullptr)
		root = cur;
	else {
		if (root->value < val) root->right = buildTree(root->right, val);
		else if (root->value > val) root->left = buildTree(root->left, val);
		else
			return root;
	}
	return root;
}
void printTree(TreeNode* root) {
	if (root) {
		printTree(root->left);
		cout << root->value << "\t";
		printTree(root->right);
	}
}
template<typename T>
class Qstack {
public:
	void push(const T& node) {
		if (que1.empty() && que2.empty()) que1.push(node);
		else if (!que1.empty()) que1.push(node);
		else if (!que2.empty()) que2.push(node);
	}
	T pop(){
		if (que1.empty() && que2.empty()) cout<<"stack is empty"<<endl;
		else if (!que1.empty()) {
			while (que1.size() > 1) {
				que2.push(que1.front());
				que1.pop();
			}
			T ret = que1.front();
			que1.pop();
			return ret;
		}
		else {
			while (que2.size() > 1) {
				que1.push(que2.front());
				que2.pop();
			}
			T ret = que2.front();
			que2.pop();
			return ret;
		}
	}
private:
	queue<T> que1;
	queue<T> que2;
};
template<typename T> int len(const T& arr) {
	return (int)sizeof(arr) / sizeof(arr[0]);
}
template<typename T> void exchange(T arr[], int a, int b) {
	arr[a] = arr[a] ^ arr[b];
	arr[b] = arr[a] ^ arr[b];
	arr[a] = arr[a] ^ arr[b];
}
void count(int arr[], int len) {
	if (len < 2) return;
	int maxNum = arr[0];
	for (int i = 1; i < len; ++i)
		if (arr[i] > maxNum) maxNum = arr[i];
	int* t = new int[maxNum + 1]();//假设数均不是负数
	for (int i = 0; i < len; ++i) t[arr[i]]++;
	int j = 0;
	for (int i = 0; i <= maxNum; ++i)
		while (int count = t[i] > 0) {
			arr[j++] = i;
			t[i]--;
		}
	delete[]t;
}
void barral(int arr[], int len, int size) {
	if (len < 2) return;
	int minV = arr[0], maxV = arr[0];
	for (int i = 1; i < len; ++i) {
		if (arr[i] < minV) minV = arr[i];
		if (arr[i] > maxV) maxV = arr[i];
	}
	if (minV == maxV) return;
	int barAmount = (maxV-minV+1)/size;
	vector<vector<int>> barral(barAmount+1, vector<int>());
	for (int i = 0; i < len; ++i) {
		int barId = arr[i]/3;
		barral[barId].emplace_back(arr[i]);
	}
	for (int i = 0; i < barral.size(); ++i)
		sort(barral[i].begin(), barral[i].end());
	int index = 0;
	for (auto v : barral)
		if (!v.empty())
			for (auto mel : v) arr[index++] = mel;
}

struct ListNode {
	ListNode* next = nullptr;
	int val;
	ListNode(int value):val(value){}
};
ListNode* addListNode(ListNode* head, int val) {
	if (head == nullptr) return new ListNode(val);
	ListNode* cur = head;
	while (cur->next) cur = cur->next;
	cur->next = new ListNode(val);
	return head;
}

TreeNode* midTraversal(TreeNode* root, int& count, int& k) {
	TreeNode* cur = nullptr;
	if (root) {
		cur = midTraversal(root->left, count, k);
		if (cur) return cur;
		if(k==++count)return root;
		cur = midTraversal(root->right, count, k);
	}
	return cur;
}
TreeNode* getKthNodeOfBinTree(TreeNode* root, int k) {
	if (root == nullptr || k <= 0) return nullptr;
	int count = 0;
	return midTraversal(root,count , k);
}

void stackSortStack(stack<int>& stk) {
	if (stk.empty()) return;
	stack<int> helper;
	while (!stk.empty()) {
		int top = stk.top();
		stk.pop();
		if (helper.empty()) helper.push(top);
		else {
			while (!helper.empty()&&top < helper.top()) {
				stk.push(helper.top());
				helper.pop();
			}
			helper.push(top);
		}
	}
	while (!helper.empty()) {
		cout << helper.top();
		helper.pop();
	}
}
int main(int argc, char** argv) {
	//注释：     先CTRL+K，然后CTRL+C
	//取消注释： 先CTRL + K，然后CTRL + U
	//设置随机种子srand(time(0))
	//获取[m,n]间的随机数 rand()%(n-m+1)+m
	//未知类型打印cout << typeid(T).name() << endl;
	//浮点数的比较尽量不用==和!=，而用fabs(a-b)>=EPSILON 或 fabs(a-b)<=EPSILON
	//float的精度是7位（即科学记数法7位有效数字），EPSILON=1e-6;double是16位,EPSILON=1e-15;
	//长循环作为内循环能减少CPU调度，提高运行效率

	//unordered_map<int, int> m;
	//vector<int> v;
	//for (int i = 0; i < 6; i++)
	//v.push_back(i);
	//auto rlast = reverse_iterator<vector<int>::iterator>(v.begin());
	//auto rfirst = reverse_iterator<vector<int>::iterator>(v.end());
	//cout << *v.begin() << " \t" << *prev(v.end())<< *rfirst<<endl;
	//reverse(v.begin(), v.end());
	//cout << *next(v.begin(), 1) << endl;
	//int *p = new int[10]();
	//cout << p[9];
	/*vector<bool> v(20, false);
	cout << sizeof(v) << "\t" << v.size() << "\t" << v.capacity();
	cout << endl;
	vector<bool> b;
	b.reserve(20);
	cout << sizeof(b) << "\t" << b.size() << "\t" << b.capacity();*/
	/*vector<int> v(10);
	for (auto i : v)
	cout << i;*/
	/*char szOrbits[] = "38414.4 29.53";
	char* pEnd;
	double d1, d2;
	cout << &d1 << endl;
	d1 = strtod(szOrbits, &pEnd);
	int d3 = d1;
	cout << &d3 << endl;
	printf("%d\n", d3);
	d2 = strtod(pEnd, NULL);
	printf("%.2f\n", d2);
	printf("The moon completes %.2f orbits per Earth year.\n", d1 / d2);*/
	//string s = "192.168.0.1..1/root/a.b/c end";
	//char* delim = "./ ";
	//char* context;
	//char* pch;
	//char* end;
	//pch=strtok_s(const_cast<char*>(s.c_str()), delim,&context);
	//while (pch)
	//{
	//	double d = strtod(pch, &end);
	//	//if(!end)
	//	cout << *end << "\t";
	//		cout << d << endl;
	//	pch=strtok_s(NULL, delim, &context);
	//}
	/*string s = "))()([)][()[]])";
	bool flag = s.end() == end(s);
	cout << flag << endl;
	system("pause");*/
	/*vector<vector<int>> vv{ {} };
	cout << vv.size() << endl;
	cout << vv[0].empty() << endl;*/
	//auto start = clock();
	//cout << clock() - start << endl;
	/*vector<int> money = { 1,2,5,10,20,50,100,200 };
	int amount = 169;
	vector<vector<int>> vret;
	double start = clock();
	cout << getSpecificAmount(money, amount,vret,10) << endl;
	cout << clock() - start << endl;
	for (auto c : vret) {
	for (auto i : c)
	cout << i << "\t";
	cout << endl;
	}*/
	/*stringstream ss;
	char c[] = { '1','2','3','\0' };
	string s = c;
	cout <<s<< endl;*/
	//string s = "1326%acm#136";
	//int left, right;
	//	for (int i = 0; i < s.size(); ++i)
	//		if (s[i] == '%')
	//			left = i;
	//	for(int i=left+1;i<s.size();++i)
	//		if (s[i] == '#') {
	//			right = i;
	//			break;
	//		}
	//	string t;
	//	for (int i = s[left - 1]-'0'; i > 0; --i)
	//		t += s.substr(left + 1, right - left-1);
	//	cout << s<<endl;
	/*vector<int> v{ 1,2,3,4,5,6 };
	std::vector<std::thread> threads;
	threads.push_back(std::thread(printArray,v));
	threads.push_back(std::thread(printArray, v));
	threads.push_back(std::thread(printArray, v));
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));*/
	/*vector<int> v = { 1,1,1,1,1,2,3,6,6,6,7,8,8,8,8,10,10,10,16,16,16};
	int k = 1;
	int n = Duplicates_ks(v,k);
	for (int i = 0; i < n; ++i)
	cout << v[i];*/
	//std::vector<std::thread> threads;
	//threads.emplace_back(thread(SingletonForHungry::getInstance));
	//threads.emplace_back(thread(SingletonForHungry::getInstance));
	//threads.emplace_back(thread(SingletonForHungry::getInstance));
	//for (auto &t : threads)
	//	t.join();
	//I need success;
	/*string s = "3[abc]c2[3[a]b]";
	cout << decodeStr(s) << endl;*/
	/*vector<int> v = { 3,6,1,5,4,2 };
	TreePNode* root = nullptr;
	for (auto i : v) root = buildTree(root, i);
	cout << getNext(root)->value << endl;*/
	/*float a = 0.9999999;
	float b = 0.9999998;
	float c = 0.0000012;
	assert(IsEqual(a - b, c, ABSOLUTE_ERROR));*/
	
	/*stringstream ss;
	ostream os(ss.rdbuf());
	istream is(ss.rdbuf());
	os << "hello,world!";
	char c;
	while (is.get(c))
		cout << c;*/
	//cout << os.str();
	vector<int> v = { 3,1,2,4,2,5,6 };
	stack<int> stk;
	for (auto i : v) stk.push(i);
	stackSortStack(stk);
	system("pause");
	return 0;
}