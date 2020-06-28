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
#include<Windows.h>

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
vector<vector<int>> LevelOrderTraversal(TreeNode* root) {
	vector<vector<int>> ret;
	if (root == nullptr)
		return ret;
	queue<TreeNode*> current, next;
	current.push(root);
	while (!current.empty())
	{
		vector<int> level;
		while (!current.empty())
		{
			TreeNode* tmp = current.front();
			current.pop();
			level.push_back(tmp->value);
			if (tmp->left)
				next.push(tmp->left);
			if (tmp->right)
				next.push(tmp->right);
		}
		ret.push_back(level);
		swap(current, next);
	}
	return ret;
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

int getSpecificSubArrayAmount(vector<int>& v, int target) {
	if (v.empty()) return 0;
	int counts = 0;
	int i = 0;
	int j = 0;
	deque<int> qMax;
	deque<int> qMin;
	while (i < v.size()) {
		while (j < v.size()) {
			while (!qMax.empty() && v[j] >= v[qMax.back()]) qMax.pop_back();
			qMax.push_back(j);
			while (!qMin.empty() && v[j] <= v[qMin.back()]) qMin.pop_back();
			qMin.push_back(j);
			if (v[qMax.front()] - v[qMin.front()]  > target) break;
			j++;
		}
		if (qMax.front() == i) qMax.pop_front();
		if (qMin.front() == i) qMin.pop_front();
		counts += j - i;
		i++;
	}
	return counts;
}

void addRec(vector<vector<int>>& vv, vector<bool>& flag) {
	int a, b, x, y;
	//scanf(" %d %d %d %d",&a,&b,&x,&y);
	cin >> a >> b >> x >> y;
	vector<int> rec = { a,b,x,y };
	vv.emplace_back(rec);
	flag.emplace_back(true);
}

void rmRec(vector<vector<int>>& vv, vector<bool>& flag) {
	int which;
	//scanf(" %d", &which);
	cin >> which;
	//vv.erase(remove(vv.begin(),vv.end(),vv[which-1]));
	flag[which - 1] = false;
}

void query(vector<vector<int>>& v, vector<bool>& flag) {
	int a, b, x, y;
	//scanf(" %d %d %d %d",&a,&b,&x,&y);
	cin >> a >> b >> x >> y;
	int counts = 0;
	for (int i = 0; i<v.size(); ++i) {
		if (flag[i] == false) continue;
		if (v[i][1]>y || v[i][3]<b || v[i][2]<a || v[i][0]>x)
			continue;
		counts++;
	}
	printf("%d\n", counts);
}

class MyString {
public:
	explicit MyString(){
		cout << "call ::nonarg constructor()" << this<< endl;
	}
	explicit MyString(char* string):data(string){
		if (string == nullptr) return;
		int size = 0;
		char* p = string;
		while (*p != '\0') {
			++size;
			++p;
		}
		size_t = size + 1;
		data = new char[size_t];
		if (data == nullptr) return;
		memcpy(data, string, size_t);
		cout << "call ::constructor()" << this<<endl;
	}
	explicit MyString(const MyString& string) {//禁止这样拷贝构造MyString sss = ss;
		if (string.data == nullptr) return;
		int size = 0;
		char* p = string.data;
		while (*p != '\0') {
			++size;
			++p;
		}
		size_t = size + 1;
		data = new char[size_t];
		if (data == nullptr) return;
		memcpy(data, string.data, size_t);
		cout << "call ::copy constructor()" <<this<< endl;
	}
	~MyString() {
		if (data != nullptr)	delete data;
		data = nullptr;
		size_t = 0;
		cout << "call ::destructor()" << this<< endl;
	}
	MyString(MyString&& string)noexcept:data(string.data){//需要移动语义，所以不能为const
		string.data = nullptr;
		cout << "move copy constructor()" << this << endl;
	}
	const MyString& operator=(const MyString& string) {//异常安全做法,在内存不够抛出bad_alloc时不会改变原data的内容
		if (&string == this) return *this;
		MyString nonConst(string);//string 是 const 常量，不能更改，所以借助中间值
		char* tmp = nonConst.data;//这里由于是指针，不存在拷贝，所以std::move()可用可不用
		nonConst.data = data;
		data = tmp;
		size_t = string.size_t;
		cout << "call ::assightment" <<  endl;
		return *this;
		//退出前将析构nonConst中间变量，即delete nonConst.data指向的内存，这里交换后就是原来data所指的区域
	}
	const MyString& operator=(MyString&& string)noexcept {
		if (&string == this) return *this;
		delete data;
		data = string.data;
		string.data = nullptr;
		cout << "call ::move assightment" << endl;
		return *this;
	}
	char* getString() {
		return data;
	}
private:
	char* data = nullptr;
	uint64_t size_t = 0;
};

class ReversePolishNotation {
public:
	ReversePolishNotation(string s) {
		if (!s.empty()) {
			transferToRePolish(s);
			computeReversePolish();
		}
	}
	//计算逆波兰表达式的值，栈的经典应用
	void computeReversePolish() {
		stack<float> stk;
		for (auto s : v) {
			if (!isOperator(s)) {
				stk.emplace(stof(s));
			}
			else if (stk.size() == 1) {
				cout << s;
			}
			else {
				float y = stk.top();
				stk.pop();
				float x = stk.top();
				stk.pop();
				float num;
				if (s == "+") num = x + y;
				else if (s == "-") num = x - y;
				else if (s == "*") num = x*y;
				else num = x / y;
				stk.emplace(num);
			}
		}
		cout << stk.top() << endl;
	}
private:
	//中缀表达式转换为逆波兰表达式，也是栈的经典应用
	//遍历到完整数字，追加到逆波兰表达式式尾
	//遍历到左括号入栈，遇到右括号，弹栈直到遇到左括号，把弹出的元素追加到逆波兰表达式式尾
	//遍历操作符ch，如果它的优先级小于等于栈顶操作符的优先级，则弹栈，把弹出的元素追加到逆波兰表达式式尾，最后把该操作符ch压栈，维持栈顶优先级最高
	//遍历结束时，弹出所有的元素，并依次追加到逆波兰表达式式尾
	void transferToRePolish(const string& s) {
		int size = s.size();
		stack<char> stk;
		int i = 0;
		while (i < size) {
			if ((s[i] >= '0'&&s[i] <= '9') || s[i] == '.') {
				int j = i;
				while (s[i] >= '0'&&s[i] <= '9')
					i++;
				if (j - 1 >= 0 && (s[j - 1] == '-' || s[j - 1] == '+') && j - 2 >= 0 && s[j - 2] == '(') {
					stk.pop();
					v.emplace_back(s.substr(j - 1, i - j + 1));
				}
				else
					v.emplace_back(s.substr(j, i - j));
			}
			else if (s[i] == '(') {
				stk.push(s[i++]);
			}
			else if (s[i] == ')') {
				while (stk.top() != '(') {
					char ch = stk.top();
					string ss;
					ss += ch;
					v.emplace_back(ss);
					stk.pop();
				}
				stk.pop();
				i++;
			}
			else {
				if (stk.empty())
					stk.emplace(s[i]);
				else {
					while (!stk.empty() && priority(s[i]) <= priority(stk.top())) {
						v.emplace_back(string(stk.top(),1));
						stk.pop();
					}
					stk.emplace(s[i]);
				}
				i++;
			}
		}
		while (!stk.empty()) {
			char ch = stk.top();
			string ss;
			ss += ch;
			v.emplace_back(ss);
			stk.pop();
		}

		/*for (auto i : v)
			cout << i << "\t";
		cout << endl;*/
	}
	int priority(char ch) {
		switch (ch) {
		case '(': return 0;
		case '-':
		case '+': return 1;
		case '*':
		case '/': return 2;
		default: return -1;
		}
	}
	bool isOperator(const string& s) {
		return s.size() == 1 && string("+-*/").find(s[0]) != string::npos;
	}
private:
	vector<string> v;
};

const float absError = 1e-6;
bool isEquire(float a, float b) {
	return fabs(a - b) <= absError;
}
int random01p() {
	float p = 0.87;//p值任意指定
	//srand(time(nullptr));
	float randomGet = ((float)(rand() % 1000)) / 1000;
	return (randomGet < 0.87||isEquire(randomGet,p)) ? 0 : 1;
}

class Num2Chinese {
public:
	Num2Chinese(int num) {
		translater(num);
	}
private:
	void translater(int n) {
		string ret;
		bool isMinus = n < 0 ? true : false;
		if (n == 0) {
			cout << Cmap_[n] << endl;
			return;
		}
		if (n < 0) 
			n = ~n+1;//非常值得注意的是，INT_MIN用例是无法通过这条语句的，因为编译器为了防止-INT_MIN溢出，所以默认对INT_MIN取相反数失效
		if (n % 10 > 0)
			ret += Cmap_[n % 10];
		n /= 10;
		int bit = 1;
		while (n > 0) {
			if (n % 10 == 0 && n % 100 != 0) {
				if(!ret.empty())
					ret = Cmap_[n % 10] + ret;
			}
			else {
				if (n % 10 != 0) {
					ret = Bmap_[bit] + ret;
					ret = Cmap_[n % 10] + ret;
				}
			}
			bit++;
			n /= 10;
		}

		if (isMinus)
			cout << "负";
		cout << ret << endl;
		/*输出中文，特别重要!先转char* ，再printf
		const char* s = ret.c_str();
		for (int i = 0; i < ret.size(); i=i+2)
			printf("%c%c", ret[i],ret[i+1]);
		*/
	}
private:
	vector<string> Cmap_ = { "零","一","二","三","四","五","六","七","八","九" };
	vector<string> Bmap_ = { "个","十","百","千","万","十","百","千","亿","十","百" };//INT_MAX=2147483647;
	char ch = '四';
};
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

	//int* p = new int(1);
	//{
	//	shared_ptr<int> ptr(p);
	//	cout << ptr.use_count() << endl;
	//	shared_ptr<int> ptr2(ptr);
	//	cout << ptr2.use_count() << endl;
	//}
	//unique_ptr<int> uptr = make_unique<int>(1);
	//typedef void(*ptr)(int , int);
	//unique_ptr<int> uptr2(std::move(uptr));//std::move()转移了控制权后，uptr内的指针将会被置空，故无法再使用uptr了，uptr会在退出生存期的时候析构
	//*uptr2 = 1;
	//cout << *uptr << endl;
	/*{
		MyString s("123456");
		MyString ss(std::move(s));
		MyString sss;
		sss = std::move(ss);
		cout << sss.getString() << endl;
	}*/
	//int32的最小值该用INT_MIN表示，而不能用临界值-2147483648表示，因为编译器怕以后对该数进行取反后当正数使用超出了正数表示范围造成隐患https://blog.csdn.net/liuhhaiffeng/article/details/53991071
	string s = "123456";
	Num2Chinese obj(INT_MAX);
	cout << INT_MIN << "\t" << INT_MAX << endl;
	system("pause");
	return 0;
}