#include<iostream>
using namespace std;
#include<algorithm>
#include<functional>
#include<numeric>
#include<stack>
#include<queue>
#include<concurrent_priority_queue.h>
#include<array>
#include<string>
#include<vector>
#include<forward_list>
#include<list>
#include<deque>
#include<unordered_map>
#include<unordered_set>
#include<bitset>
#include<time.h>


//第一遍
namespace Array {
	//March_3 移除已排序重复的元素
	//时间复杂度O(n)，空间复杂度O(1)
	int  removeDuplicates(vector<int>& v) {
		if (v.size()<=1) return v.size();
		int index = 1;
		for (int i = 1; i < v.size(); ++i)
			if (v[index - 1] != v[i])
				v[index++] = v[i];
		return index;
	}
	//元素最多重复两次
	//时间复杂度O(n)，空间复杂度O(1)
	int Duplicate_twice(vector<int>& v) {
		if (v.size() <= 2) return v.size();
		int index = 2;
		for (int i = 2; i < v.size(); ++i)
			if (v[i] != v[index - 2])
				v[index++] = v[i];
		return index;
	}
	//通式，允许k次重复
	int Duplicates_ks(vector<int>& v, int k) {
		if (v.size() <= k) return v.size();
		int index = k;
		for (int i = k; i < v.size(); ++i)
			if (v[i] != v[index - k])
				v[index++] = v[i];
		return index;
	}
	//Remove Element in place,Return New Length
	int rmElement(vector<int>& v, int target) {
		if (v.empty()) return v.size();
		int index = 0;
		for (int i = 0; i < v.size(); ++i)
			if (target != v[i])
				v[index++] = v[i];
		return index;
	}
	//有序旋转数组中查找元素，二分法
	//时间复杂度O(logN)，空间复杂度O(1)
	//没有重复元素
	int SearchInTheRotatedArray(const vector<int>& v,int target) {
		if (v.empty())return -1;
		int left = 0, right = v.size()-1;
		while (left <= right) {
			int mid = left + (right - left) / 2;
			if (v[mid] == target) return mid;
			if (v[left] < v[mid]) 
				if (v[left] <= target&&target < v[mid]) right = mid - 1;
				else left = mid + 1;
			else 
				if (v[mid]<target&&target<=v[right]) left = mid + 1;
				else right = mid - 1;
		}
		return -1;
	}
	//有重复元素，查找就只要判断target是否存在即可
	//逻辑和上题是一样的，唯一的不同是当A[m]>=A[l]时，那么[l,m]为递增序列的假设就不能成立了，比如[1,2,1,1,1]
	//只需将它拆成两个条件：if(A[m]>A[l])时，[l,m]还是递增序列；if(A[m]==A[l])，比如[1,2,1,1,1]，则只能表示A[l]不等于target，l++；
	bool searchInRotatedArrayWithDumplicates(const vector<int>& v, int target) {
		if (v.empty()) return -1;
		int left = 0, right = v.size() - 1;
		while (left <= right) {
			int mid = left + (right - left) / 2;
			if (v[mid] == target) return true;
			if (v[left] < v[mid])
				if (v[left] <= target&&target < v[mid]) right = mid - 1;
				else left = mid + 1;
			else if (v[left] > v[mid])
				if (v[mid] < target&& target <= v[right]) left = mid + 1;
				else right = mid - 1;
			else left++;
		}
		return false;
	}
	//给定两个有序数组，找出她们合并后的中位数，要求时间复杂度为log(M+N)
	//此问题的另一个经典描述为--找出两个有序数组中第k小的数
	//方法一：用一个计数器count，用emerge排序遍历的方式，设置两个指针PA和PB分别指向两个数组，如果*PA<*PB，则PA++,count++；
	//如果*PA>*PB，则PB++,count++；直到count==k，返回造成count=k时对应的*PA或*PB。时间复杂度为O(m+n)，空间复杂度O(1)。
	//方法二：利用伪二分法，假设数组的大小m,n均大于k/2，通过来比较A[k/2]和B[k/2]可以减半排除部分元素，假如A[k/2]==B[k/2]，
	//则找到了第k小的值；A[k/2]<B[k/2]，则A[0]-A[k/2]的数肯定小于第k小的那个数，反之B[0]-B[k/2]小于那个target，如此缩小范围。
	class Solution1 {
	public:
		double Mid_Num(vector<int>& a, vector<int>& b) {//找中位数
			int m = a.size();
			int n = b.size();
			int total = n + m;
			if (total & 0x1)
				return find_kth(a.begin(), m, b.begin(), n, total / 2 + 1)*1.0;
			else
				return (find_kth(a.begin(), m, b.begin(), n, total / 2 + 1) + find_kth(a.begin(), m, b.begin(), n, total / 2)) / 2.0;
		}
	private:
		int find_kth(vector<int>::iterator A, int m, vector<int>::iterator B, int n, int k) {//找第K小的数
			//均假设m<=n
			if (m > n) return find_kth(B, n, A, m, k);
			if (0 == m) return *(B + k - 1);
			if (1 == k) return min(*A, *B);
			int ai = min(k / 2, m);
			int bi = k - ai;
			if (*(A + ai - 1) < *(B + bi - 1)) return find_kth(A + ai, m - ai, B, n, k - ai);
			else if (*(A + ai - 1) > *(B + bi - 1)) return find_kth(A, m, B + bi, n - bi, k - bi);
			else return *(A + ai - 1);
		}
	};
	//在无序数组中找到最长的连续子序列，返回它的长度，要求时间复杂度O(N)
	//例如[100,6,2,99,5,3,4,1,98]中最长的连续子序列为[1,2,3,4,5,6]
	//方法：存入hash容器（O(N)）,往左往右寻找连续元素（O(N)）,总为O(N)
	int longestConcurrentSubArray(vector<int>& v) {
		unordered_map<int, bool> m;
		for (auto i : v) m[i] = false;
		int length = 0;
		int longest = 0;
		for (auto i : v) {
			if (v[i]) continue;
			m[i] = true;
			length = 1;
			for (int j = i + 1; m.find(j) != m.end(); ++j) {
				m[j] = true;
				++length;
			}
			for (int j = i - 1; m.find(j) != m.end(); --j) {
				m[j] = true;
				++length;
			}
			longest = max(longest, length);
		}
		return longest;
	}
	//找到数组中和等于sum的两个数
	//在hash容器中建立值和下标间的映射，再遍历寻找目标对组
	//时间复杂度为O(N)
	vector<pair<int, int>> TwoSum(vector<int>& v, int sum) {
		unordered_map<int, int> m;
		vector<pair<int, int>> retv;
		if (v.size() < 2) return retv;
		for (int i = 0; i < v.size(); ++i)
			m[v[i]] = i;
		for (int i = 0; i < v.size(); ++i) {
			int gap = sum - v[i];
			if (m.find(gap) != m.end() && m[gap] > i) {
				retv.push_back(pair<int, int>(i + 1, m[gap] + 1));
				break;
			}
		}
		return retv;
	}
	//同理可得3sum=0的码，时间复杂度为O(N)
	vector<vector<int>> ThreeSum(vector<int>& v, int sum) {
		unordered_map<int, int> m;
		vector<vector<int>> retv;
		if (v.size() < 3)
			return retv;
		for (int i = 0; i < v.size(); ++i)
			m[v[i]] = i;
		int twoSum;
		vector<int> tmpv;
		for (int i = 0; i < v.size(); ++i) {
			twoSum = sum - v[i];
			for (int j = i + 1; j < v.size(); ++j)
				tmpv.push_back(v[j]);
			vector<pair<int, int>> set = TwoSum(tmpv, twoSum);
			if (set.empty())
				continue;
			for (auto pair : set) {
				tmpv.push_back(i + 1);
				tmpv.push_back(pair.first);
				tmpv.push_back(pair.second);
				retv.push_back(tmpv);
			}
		}
		return retv;
	}
	//closest3Sum，求最逼近某个target值的三个数的和
	//时间复杂度O(N**2)
	int closest3Sum(vector<int>& v,int target) {
		int closest = 0;
		int min_gap = INT_MAX;
		int rets;
		int sum;
		sort(v.begin(), v.end());
		for (auto a = v.begin(); a != prev(v.end(), 2); ++a) {
			auto b = next(a);
			auto c = prev(v.end());
			while (b < c) {
				sum = *a + *b + *c;
				int gap = abs(target - sum);
				if (gap < min_gap) {
					min_gap = gap;
					rets = sum;
				}
				if (sum > target) --c;
				else if (sum < target) ++b;
				else return sum;
			}
		}
		return sum;
	}
	//4sum==target
	//时间复杂度为O(N**2)
	vector<vector<int>> get4Sum(vector<int>& v, int target) {
		vector<vector<int>> retv;
		if (v.empty()) return retv;
		unordered_multimap<int, pair<int, int>> mulmap;
		for (int i = 0; i < v.size(); ++i)
			for (int j = i + 1; j < v.size(); ++j)
				mulmap.insert(make_pair(v[i]+v[j],make_pair(i,j)));
		for (auto i = mulmap.begin(); i != mulmap.end(); ++i) {
			int gap = target - i->first;
			if (mulmap.find(gap) == mulmap.end()) continue;
			auto range = mulmap.equal_range(gap);
			int a, b, c, d;
			for (auto j = range.first; j != range.second;++j) {
				a = i->second.first;
				b = i->second.second;
				c = j->second.first;
				d = j->second.second;
				if (a != c&&a != d&&b != c&&b != d) {
					vector<int> tmpv = { v[a],v[b],v[c],v[d] };
					sort(tmpv.begin(), tmpv.end());
					retv.push_back(tmpv);
				}
			}
		}
		sort(retv.begin(), retv.end());
		retv.erase(unique(retv.begin(), retv.end()), retv.end());
		return retv;
	}
	//得到比这个排列稍大的下一个排列
	//时间复杂度O(N)，空间复杂度O(1)
	class solution2 {
		void getNextPermutation(vector<int>& v) {
			nextPermutation(v.begin(), v.end());
		}
		template<typename bidiIt>
		bool nextPermutation(bidiIt first, bidiIt last) {
			auto rfirst = reverse_iterator<bidiIt>(last);
			auto rlast = reverse_iterator<bidiIt>(first);
			auto pivot = next(rfirst);
			while (pivot != rlast&&*pivot >= *prev(pivot))
				++pivot;
			if (pivot == rlast) {
				reverse(rfirst, rlast);
				return false;
			}
			auto change = find_if(rfirst, pivot, bind1st(less<int>(), *pivot));
			swap(*change, *pivot);
			reverse(rfirst, pivot);
			return true;
		}
	};
	//More Advanced Solution:康拓编解码
	//康拓解码(这个数是第几个排列)和康拓解码(第k个排列的数是多少)
	class solution3 {
	public:
		vector<int> getKthPermutation(int n, int k) {
			vector<int> v;
			for (int i = 1; i <= n; ++i)
				v.push_back(i);
			return kthPermutation(v, k);
		}
	private:
		vector<int> kthPermutation(vector<int> v, int k) {
			int n = v.size();
			vector<int> f(n, 0);
			f[0] = 1;
			for (int i = 1; i < n; ++i)
				f[i] = i*f[i - 1];
			if (k > f[n - 1] * n) return vector<int>{-1};//超出范围不存在则返回-1
			--k;//有k-1个比他小的数
			vector<int> vret;
			while (k != 0) {
				auto a = next(v.begin(), k / f[--n]);
				vret.push_back(*a);
				v.erase(a);
				k %= f[n];
			}
			vret.insert(vret.end(), v.begin(), v.end());
			return vret;
		}
	};
	//判断是否是有效数独，空格是用'.'填充表示的
	//由于只要检查9行9列9个九宫格，都不会很复杂，所以复杂度可以忽略
	class solution4 {
	public:
		bool isValidSudoku(vector<vector<char>>& v) {
			bool used[9];
			for (int i = 0; i < 9; ++i) {
				fill(used, used + 9, false);
				for (int j = 0; j < 9; ++j) 
					if (!check(v[i][j], used))//检查行
						return false;
				fill(used, used + 9, false);
				for (int j = 0; j < 9; ++j)
					if (!check(v[j][i], used))//检查列
						return false;
			}
			for (int r = 0; r < 3; ++r) //检查九宫格
				for (int c = 0; c < 3; ++c) {
					fill(used, used + 9, false);
					for (int i = 3 * r; i < 3 * r + 3; ++i)
						for (int j = 3 * c; j < 3 * c + 3; ++j)
							if (!check(v[i][j], used)) return false;
				}
			return true;
		}
	private:
		bool check(char c, bool used[]) {
			if (c == '.')return true;
			if (used[c - '1'])return false;
			return used[c - '1'] = true;

		}
	};
	//直方图接水问题
	//时间复杂度O(N)，空间复杂度O(1)
	int trappingRain(vector<int>& v) {
		int max = 0;
		for (int i = 1; i < v.size(); ++i)
			if (v[i] > v[max])
				max = i;	
		int water = 0, peak = 0;
		for (int i = 0; i < max; ++i) {
			if (v[i] > peak)
				peak = v[i];
			else
				water += peak - v[i];
		}
		peak = 0;
		for (int j = v.size() - 1; j > max; --j) {
			if (v[j] > peak)
				peak = v[j];
			else
				water += peak - v[j];
		}
		return water;
	}
	//方阵旋转问题
	//把方阵顺时针旋转90度：先对角线翻转，再沿着矩阵的垂直中线左右翻转
	//把方阵逆时针旋转90度：也是先对角线翻转，但是第二步是沿着水平中线上下翻转
	void rotateSquareI(vector<vector<int>>& v) {
		int n = v[0].size();
		//对角线翻转
		for (int i = 0; i < n; ++i)
			for (int j = i + 1; j < n; ++j)
				swap(v[i][j], v[j][i]);
		//顺时针旋转90度：左右翻转
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n / 2; ++j)
				swap(v[i][j], v[i][n - i - 1]);
	}
	void rotateSquareII(vector<vector<int>>& v) {
		int n = v[0].size();
		//对角线翻转
		for (int i = 0; i < n; ++i)
			for (int j = i + 1; j < n; ++j)
				swap(v[i][j], v[j][i]);
		//逆时针旋转90度，上下翻转
		for (int i = 0; i < n / 2; ++i)
			for (int j = 0; j < n; ++j)
				swap(v[i][j], v[n - 1 - i][j]);
	}
	//大数加一问题
	vector<int>& plusOne(vector<int> v) {
		int c = 1;//进位
		for (auto i = v.rbegin(); i != v.rend(); ++i) {
			*i += c;
			c = *i / 10;
			*i %= 10;
		}
		if (c > 0) v.insert(v.begin(), c);
		return v;
	}
	//斐波拉契楼梯,f(n)=f(n-1)+f(n-2)
	int climbStairs(int n) {
		if (n < 1)
			return 0;
		int pre = 0;
		int cur = 1;
		for (int i = 1; i <= n; ++i) {
			int tmp = cur+pre;
			pre = cur;
			cur=tmp;
		}
		return cur;
	}
	//格雷码
	class solution5 {
	public:
		vector<int> grayCode(int n) {
			size_t size = 1 << n;
			vector<int> v;
			v.reserve(size);
			for (int i = 0; i < size; ++i)
				v.push_back(nextGrayCode(i));
			return v;
		}
	private:
		unsigned int nextGrayCode(unsigned int n) {
			return n ^ (n >> 1);
		}
	};
	//将矩阵含0的横纵置零
	//时间复杂度O(m*n)，空间复杂度(m+n)
	void setMatrixZerosI(vector<vector<int>>& v) {
		const int m = v.size();
		const int n = v[0].size();
		vector<bool> row(m,false);
		vector<bool> col(n,false);
		for(int i=0;i<m;++i)
			for(int j=0;j<m;++j)
				if (v[i][j] == 0) 
					row[i] = col[j] = true;
		for (int i = 0; i < m; ++i)
			if (row[i])
				fill(&v[i][0], &v[i][n-1],0);
		for (int j = 0; j < n; ++j)
			if (col[j])
				for (int i = 0; i < m; ++i)
					v[i][j] = 0;
	}
	//常数空间消耗
	//Solution :复用第一行和第一列
	//时间复杂度O(m*n)，空间复杂度O(1)
	void setMatrixzerosII(vector<vector<int>>& v) {
		bool firstRowHasZero = false;
		bool firstColHasZero = false;
		const int m = v.size();
		const int n = v[0].size();
		for (int j = 0; j < n; ++j)
			if (v[0][j] == 0) {
				firstRowHasZero = true; break;
			}
		for (int i = 0; i < m; ++i) 
			if (v[i][0] == 0) {
				firstColHasZero = true; break;
			}
		for (int i = 1; i < m; ++i)
			for (int j = 1; j < n; ++j)
				if (v[i][j] == 0)
					v[0][j] = v[i][0] = 0;
		for (int i = 1; i < m; ++i)
			for (int j = 1; j < n; ++j)
				if (v[i][0] == 0 || v[0][j] == 0) v[i][j] = 0;
		if (firstRowHasZero)
			for (int j = 0; j < n; ++j) v[0][j] = 0;
		if (firstColHasZero)
			for (int i = 0; i < m; ++i) v[i][0] = 0;
	}
	//gas station
	//时间复杂度O(N),空间复杂度O(1)
	int gasStation(vector<int>& gas, vector<int>& cost) {
		int amount = 0;
		int sum = 0;
		int j = -1;
		for (int i = 0; i < gas.size(); ++i) {
			sum += gas[i] - cost[i];
			amount += gas[i] - cost[i];
			if (sum < 0) {
				j = i;
				sum = 0;
			}
		}
		return amount >= 0 ? j + 1 : -1;
	}
	//candy
	//时间复杂度O(n)，空间复杂度O(n)
	int getMinCandy(vector<int>& v) {
		int n = v.size();
		vector<int> vret(n,1);
		for (int i = 1; i < n; ++i)
			if (v[i] > v[i - 1])
				vret[i] = vret[i - 1] + 1;
		for (int i = n - 1; i > 0; --i)
			if (v[i - 1] > v[i])
				vret[i - 1] = max(vret[i]+1, vret[i - 1]);
		return accumulate(vret.begin(), vret.end(), 0);
	}
	//Find Single Number,从重复2次的数中找到唯一出现一次的数
	//异或，出现偶数次都可以清零
	//时间复杂度O(n)，空间复杂度O(1)
	int FindSingleNumberr(vector<int>& v) {
		size_t ret = 0;
		for (auto i : v)
			ret ^= i;
		return ret;
	}
	//Find Single Number,从重复3次的数中找到唯一出现一次的数
	//时间复杂度O(n)，空间复杂度O(1)
	int findSinglenumber(const vector<int>& v) {
		int ret = 0;
		for (int i = 0; i < 32; ++i) {
			int sum = 0;
			for (auto j : v)
				sum += (j >> i) & 1;
			ret |= (sum % 3) << i;
		}
		return ret;
	}
}
namespace LinkList
{
	struct ListNode {//定义链表节点
		int val;
		ListNode* next = nullptr;
		ListNode(int v) :val(v) {}
	};
	//addTwoNumber
	//For example:Input:(2->4->3)+(5->6->4);Output:7->0->8
	ListNode* AddTwoNumber(ListNode* first, ListNode* second) {
		ListNode head(-1);
		ListNode* prev = &head;
		int carry = 0;
		ListNode* pa = first, *pb = second;
		while(pa != nullptr || pb != nullptr) {
			const int ai = pa == nullptr ? 0 : pa->val;
			const int bi = pb == nullptr ? 0 : pb->val;
			const int ri = (ai + bi + carry) % 10;
			carry = (ai + bi + carry) / 10;
			prev->next = new ListNode(ri);
			prev = prev->next;
			pa = pa == nullptr ? nullptr : pa->next;
			pb = pb == nullptr ? nullptr : pb->next;
		}
		if (carry > 0)
			prev->next = new ListNode(carry);
		return head.next;
	}
	//Reverse Linked List II
	//Reverse a linked list from position m to n. Do it in-place and in one-pass.
	//For example : Given 1->2->3->4->5->nullptr, m = 2 and n = 4,
	//return 1->4->3->2->5->nullptr.
	//Note : Given m, n satisfy the following condition : 1 ≤ m ≤ n ≤ length of list.
	//时间复杂度O(n)，空间复杂度O(1)
	ListNode* ReverseLinkedListII(ListNode* head, int m, int n) {
		if (head == nullptr || m == n)
			return head;
		ListNode* prev = head;
		for (int i = 1; i < m - 1; ++i)
			prev = prev->next;
		ListNode* head2 = prev;
		prev = head2->next;
		ListNode* cur = prev->next;

		for (int i = m+1; i <= n; ++i) {//头插法
			prev->next = cur->next;
			cur->next = head2->next;
			head2->next = cur;
			cur = prev->next;
		}
		return head;
	}
	//Given a linked list and a value x, partition it 
	//such that all nodes less than x come before nodes greater than or equal to x.
	//For example, Given 1->4->3->2->5->2 and x = 3, return 1->2->2->4->3->5.
	ListNode* PartitionList(ListNode* head, int val) {
		if (head == nullptr)
			return head;
		ListNode first_head(-1);
		ListNode second_head(-1);

		ListNode* first_cur = &first_head;
		ListNode* second_cur = &second_head;
		for (ListNode* cur = head; cur; cur = cur->next) {
			if (cur->val < val) {
				first_cur->next = cur;
				first_cur = cur;
			}
			else {
				second_cur->next = cur;
				second_cur = cur;
			}
		}
		first_cur->next = second_head.next;
		second_cur->next = nullptr;

		return first_head.next;
	}
	//Remove Duplicates from Sorted LinkedList
	//时间复杂度O(n)，空间复杂度O(1)
	ListNode* RemoveDuplicates(ListNode* head) {
		if (head == nullptr)
			return head;
		ListNode* prev = head;
		for (ListNode* cur = prev->next; cur; cur = prev->next) {
			if (prev->val == cur->val) {
				prev->next = cur->next;
				delete cur;
			}
			else {
				prev = cur;
			}
		}
		return head;
	}
	//Remove Duplicates from Sorted LinkedListII
	//时间复杂度O(n)，空间复杂度O(1)
	ListNode* RemoveDuplicatesII(ListNode* head) {
		ListNode bg(-1);
		ListNode* prev = &bg;
		ListNode* cur = head;
		while (cur != nullptr) {
			bool flag_dup = false;
			while (cur->next != nullptr && cur->val == cur->next->val) {
				ListNode* tmp = cur;
				cur = cur->next;
				delete tmp;
				flag_dup = true;
			}
			if (flag_dup) {
				ListNode* tmp = cur;
				cur = cur->next;
				delete tmp;
				continue;
			}
			prev->next = cur;
			prev = cur;
			cur = cur->next;
		}
		prev->next = cur;
		return bg.next;
	}
	//Rotate Linked List
	//Given a list, rotate the list to the right by k places, where k is non-negative.
	//时间复杂度O(n)，空间复杂度O(1)
	ListNode* RotateList(ListNode* head, int k) {
		if (head == nullptr || k == 0)
			return head;
		int length = 1;
		ListNode* cur = head;
		while (cur->next) {
			++length;
			cur = cur->next;
		}
		int span = length - k%length;
		cur->next = head;//首尾相连闭环
		for (int i = 0; i < span; ++i)
			cur = cur->next;
		head = cur->next;
		cur->next = nullptr;//断环
		return head;

	}
	//Remove the Nth from End of Linked List
	//For example, Given linked list: 1->2->3->4->5, and n = 2.
	//After removing the second node from the end, the linked list becomes 1->2->3->5.
	// Given n will always be valid. Try to do this in one pass.
	ListNode* RemoveNthFromEnd(ListNode* head, int n) {
		ListNode hret(-1);//建立领衔节点，应对可能存在的删除头结点
		hret.next = head;
		ListNode* p = &hret, *q = &hret;
		for (int i = 0; i < n; ++i)
			q = q->next;
		while (q->next) {
			q = q->next;
			p = p->next;
		}
		ListNode* tmp = p->next;
		p->next = tmp->next;
		delete tmp;
		return head;
	}
	//Swap Nodes in pair
	//You may not modify the values in the list, only can changed the pointer *next.
	//For example, Given 1->2->3->4, you should return the list as 2->1->4->3.
	ListNode* SwapNodesInPair(ListNode* head) {
		if (head != nullptr || head->next == nullptr) return head;
		ListNode bg(-1);
		bg.next = head;
		ListNode* prev = &bg;
		ListNode* cur = prev->next;
		ListNode* next = cur->next;
		while (next) {
			prev->next = next;
			cur->next = next->next;
			next->next = cur;

			prev = cur;
			cur = cur->next;
			next = cur ? cur->next : nullptr;
		}
		return bg.next;
	}
	//  Reverse the nodes of a linked list k at a time and return its modified list.
	// If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.
	//You may not alter the values in the nodes, only nodes itself may be changed.
	//For example, Given this linked list: 1->2->3->4->5.
	//For k = 2, you should return: 2->1->4->3->5.
	//For k = 3, you should return: 3->2->1->4->5.
	//时间复杂度O(n)，空间复杂度O(1)
	class Solution {
	public:
		ListNode* ReverseKGroups(ListNode* head, int k) {
			if (head == nullptr || k < 2)
				return head;
			ListNode bg(-1);
			ListNode* prev = &bg;
			bg.next = head;
			ListNode* end = head;
			for (; end; end = prev->next) {
				for (int i = 1; i < k&&end; ++i)
					end = end->next;
				if (end == nullptr)
					break;
				prev = reverseKGroup(prev, prev->next, end);
			}
			return bg.next;
		}
	private:
		//prev指向区间的前一个节点，[begin,end]是待reverse闭区间.
		ListNode* reverseKGroup(ListNode* prev, ListNode* begin, ListNode* end) {
			ListNode* head = prev, *next_end = end->next;
			prev = begin;
			ListNode* cur = prev->next;
			while (cur != next_end) {
				prev->next = cur->next;
				cur->next = head->next;
				head->next = cur;
				cur = prev->next;
			}
			return prev;//这个prev形参和实参prev并不是同一个指针，所以要更新实参的prev还是得返回更新
		}
	};
	//Copy Linked List with random pointer
	//时间复杂度O(n)，空间复杂度O(1)
	class RandomPointer {
		struct ListNode {
			int val;
			ListNode* next = nullptr;
			ListNode* random = nullptr;
			ListNode(int value) :val(value) {}
		};
		ListNode* CopyRandomPointer(ListNode* head) {
			for (ListNode* cur = head; cur;) {
				ListNode tmp(cur->val);
				tmp.next = cur->next;
				cur->next = &tmp;
				cur = tmp.next;
			}
			ListNode* cur = head;
			while( cur) {
				if (cur->random)//注意求cur->random->next时，先得判断cur->random的有效性
					cur->next->random = cur->random->next;
				cur = cur->next->next;
			}
			ListNode* head2 = head->next;
			cur = head;
			ListNode* next = cur->next;
			while(next) {
				cur->next = next->next;
				cur = next;
				next = next->next;
			}
			return head2;
		}
	};
	//Linked List Cycle,Is Exiting
	//时间复杂度O(n)，空间复杂度O(1)
	bool LinkedListCycle(ListNode* head) {
		ListNode* slow = head;
		ListNode* fast = head;
		while (fast && fast->next) {
			slow = slow->next;
			fast = fast->next->next;
			if (slow == fast)
				return true;
		}
		return false;
	}
	//Linked List Cycle II,Locate the begining node of cycle.
	//时间复杂度O(n)，空间复杂度O(1)	
	ListNode* LinkedListCycleII(ListNode* head) {
		ListNode* slow = head;
		ListNode* fast = head;
		while (fast && fast->next) {
			slow = slow->next;
			fast = fast->next->next;
			if (slow == fast) {
				ListNode* cur = head;
				while (cur != slow) {
					cur = cur->next;
					slow = slow->next;
				}
				return cur;
			}
		}
		return nullptr;
	}
	//Reorder Linked List
	//For example, Given {1,2,3,4,5,6}, reorder it to {1,6,2,5,3,4}.
	//You must do this in-place without altering the nodes’ values.
	class Solution1 {
	public:
		ListNode* ReorderLinkedList(ListNode* head) {
			if (head == nullptr || head->next == nullptr)
				return head;
			ListNode* slow = head;
			ListNode* fast = head;
			while (fast && fast->next) {
				slow = slow->next;
				fast = fast->next->next;
			}
			ListNode* second = slow->next;
			slow->next = nullptr;
			ListNode* first = head;
			second = Reverse(second);

			for (; first->next;) {
				ListNode* tmp = first->next;
				first->next = second;
				second = second->next;
				first->next->next = tmp;
				first = tmp;
			}
			first->next = second;
			return head;
		}
	private:
		ListNode* Reverse(ListNode* head) {
			if (head == nullptr || head->next == nullptr)
				return head;
			ListNode* prev = head;
			ListNode* cur = head->next;
			ListNode* next = cur ? cur->next : nullptr;
			for (; cur;) {
				cur->next = prev;
				prev = cur;
				cur = next;
				next = cur ? cur->next : nullptr;
			}
			head->next = nullptr;
			return prev;
		}
	};
	//LRU缓存，节省内存的同时提高对文件的访问速度
	class LRUCache {
	private:
		struct CacheNode
		{
			int key;//适用于任何类型的主键，为了简化，此处为int
			int val;//可以为任何类型的class或struct，为了简化，此处为int
			CacheNode(int k, int v) :key(k), val(v) {}
		};
		list<CacheNode> CacheList;
		unordered_map<int, list<CacheNode>::iterator> CacheMap;
		int capacity;
	public:
		LRUCache(int capacity) :capacity(capacity) {}
		int get(int key) {
			if (CacheMap.find(key) == CacheMap.end())
				return -1;
			else {
				CacheList.splice(CacheList.begin(), CacheList, CacheMap[key]);
				CacheMap[key] = CacheList.begin();
				return CacheList.begin()->val;
			}
		}
		void set(int key, int value) {
			if (CacheMap.find(key) == CacheMap.end()) {
				if (CacheList.size() == capacity) {
					CacheMap.erase(CacheList.back().key);
					CacheList.pop_back();
				}
				CacheList.push_front(CacheNode(key, value));
				CacheMap[key] = CacheList.begin();
			}
			else {
				CacheMap[key]->val = value;
				CacheList.splice(CacheList.begin(), CacheList, CacheMap[key]);
				CacheMap[key] = CacheList.begin();
			}
		}
	};
}
namespace STR {
	//partion大小写字符，小的在前大的在后，且相对位置不变，常数空间
	//发现规律：扫描字符串，大的在小的的前面则交换，扫描次数等于大写字符的总数
	string partionLetters(string& s) {
		if (s.size() < 2) return s;
		int n = 0;
		for (auto c : s)
			if (c < 'a') ++n;
		for (int i = 0; i<n; ++i)
			for (int j = 0; j < s.size() - 1; ++j)
				if (s[j] < 'a' && s[j + 1] >= 'a')
					swap(s[j], s[j + 1]);
		return s;
	}
		//Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
		//时间复杂度O(n)，空间复杂度O(1)
	bool isParlindrome(string& s) {
			if (s.length() == 0)
				return true;
			transform(s.begin(), s.end(), s.begin(), ::tolower);
			auto left = s.begin();
			auto right = prev(s.end());
			while (left < right) {
				if (!::isalnum(*left)) ++left;
				else if (!::isalnum(*right)) --right;
				else if (*left != *right) return false;
				else {
					++left; --right;
				}
			}
			return true;
		}
	//Implement strStr()
	//法一：暴力求解
	//时间复杂度O(m*n)，空间复杂度O(1)
	int strStr(string& haystack, string& needle) {
		if (haystack.empty()|| needle.empty()) return -1;
		int N = haystack.size() - needle.size() + 1;
		for (int i = 0; i < N; ++i) {
			int j = i;
			int k = 0;
			while (j<haystack.size() && k < needle.size() && haystack[j] == needle[k]) {
				++j; ++k;
			}
			if (k == needle.size())
				return i;
		}
		return -1;
	}
	//法二：KMP模式匹配算法
	//时间复杂度O(m+n)，空间复杂度(m)
	class KMP {
	public:
		int ImpKMP(string& t, string& p) {
			if (t.length() == 0 || p.length() == 0)
				return -1;
			int* next = new int[p.length()];
			getNext(p,next);
			int i = 0, j = 0;
			while (i<t.length()&&j<p.length())
			{
				if (j == -1 || t[i] == p[j]) {
					++i;
					++j;
				}
				else {
					j = next[j];
				}
			}
			delete[]next;
			if (j == p.length())
				return i - j;
			else
				return -1;
		}
	private:
		void getNext(string& p, int next[]) {
			next[0] = -1;
			int length = p.length();
			int j = -1;
			int i = 0;
			while (i<length)
			{
				if (j == -1 || p[i] == p[j]) {
					++i;
					++j;
					next[i] = j;
				}
				else {
					j = next[j];
				}
			}
		}
	};
	//Implement atoi()，不合规输入返回零
	//时间复杂度O(n)，空间复杂度O(1)
	int ImplAtoN(const string& s) {
		if (s.length() == 0)
			return 0;
		int i = 0;
		int num = 0;
		int asign = 1;
		while (s[i] == ' ') ++i;
		if (s[i] == '-') {
			asign = -1;
			++i;
		}
		else if (s[i] == '+') {
			asign = 1;
			++i;
		}
		for (; i < s.length(); ++i) {
			if (s[i] > '9' || s[i] < '0')
				break;
			if (num > INT_MAX / 10 || (num == INT_MAX
				&& (s[i]-'0') > INT_MAX % 10)) {
				return asign == -1 ? INT_MIN : INT_MAX;	
			}
			num = num * 10 + s[i]-'0';
		}
		return num*asign;
	}
	//Add binary
	//时间复杂度O(n),空间复杂度O(1)
	string AddBinary(const string& s1, const string& s2) {
		int carry = 0;
		auto i = s1.rbegin();
		auto j = s2.rbegin();
		string ret;
		for (; i != s1.rend() || j != s2.rend();) {
			int ai = i == s1.rend() ? 0 : *i-'0';
			int bi = j == s2.rend() ? 0 : *j-'0';
			int n = ai+ bi+ carry;
			carry = n / 2;
			ret.insert(ret.begin(),n % 2 + '0');
			i = i == s1.rend() ? i : ++i;
			j = j == s2.rend() ? j : ++j;
		}
		if (carry > 0)
			ret.insert(ret.begin(), '1');
		return ret;
	}
	//Longest Parlindrome SubString
	//Manacher 's Algorithm
	//时间复杂度O(n)，空间复杂度O(n)
	class Manacher {
	public:
		string GetLongestSubstring(const string& s) {
			return ManacherAlgorithm(s);
		}
	private:
		string ManacherAlgorithm(const string& s) {
			if (s.size() < 2)
				return "";
			string t = PreProcess(s);
			vector<int> p(t.size(), 0);
			int maxR = 0, id = 0, maxRad = 0, cenPos = 0;
			for (int i = 1; i < t.size(); ++i) {
				p[i] = maxR > i ? min(p[2 * id - i], maxR - i) : 1;
				while (t[i + p[i]] == t[i - p[i]]) ++p[i];
				if (maxR < i + p[i]) {
					maxR = i + p[i];
					id = i;
				}
				if (maxRad < p[i]) {
					maxRad = p[i];
					cenPos = i;
				}
			}
			return s.substr((cenPos - maxRad) / 2, cenPos - 1);
		}
		string PreProcess(const string& s) {//配成偶数总量
			string t = "$#";
			for (int i = 0; i < s.size(); ++i) {
				t += s[i];
				t += "#";
			}
			return t;
		}
	};
	//Regular Expression Match,Implement '*' and '.'
	//时间复杂度O(n)，空间复杂度O(1)
	class RegexImpl {
	public:
		bool isMatch(const string& s, const string& p) {
			return isMatch(s.c_str(), p.c_str());
		}
	private:
		bool isMatch(const char* s, const char* p) {
			if (*p == '\0')
				return *s == '\0';
			if (*(p + 1) != '*') {
				if (*p == *s || (*p == '.' && *s != '\0'))
					return isMatch(s + 1, p + 1);
				else
					return false;
			}
			else {
				while (*p == *s || (*p == '.' && *s != '\0')) {
					if (isMatch(s, p + 2))
						return true;
					s++;
				}
				return isMatch(s, p + 2);

			}
		}
	};
		//WildCard Match
		//'?' Matches any single character. '*' Matches any sequence of characters (including the empty sequence).
		//时间复杂度O(m*n)，空间复杂度O(1)ps:https://shmilyaw-hotmail-com.iteye.com/blog/2154716
		bool WildCardMatch(const string& str, const string& ptn) {
			int p = 0, s = 0, match = 0, crashID = -1;
			while (s<str.length())
			{
				if ((p < ptn.length() && ptn[p] == '?') || str[s] == ptn[p]) {
					++p; ++s;
				}
				else if (p<ptn.length()&&ptn[p] == '*') {
					crashID = p;
					match = s;
					++p;
				}
				else if (crashID != -1) {
					p = crashID + 1;
					++match;
					s = match;
				}
				else
					return false;
			}
			while (p<ptn.length() && ptn[p] == '*')
				++p;
			if (p == ptn.length())
				return true;
			else
				return false;
		}
		//Longest Common prefix
		//时间复杂度O(m*n)，空间复杂度O(1)
		string LongestCommonPrefix(vector<string>& vstr) {
			for (int i = 0; i < vstr[0].size(); ++i)
				for (int j = 1; j < vstr.size(); ++j) 
					if (vstr[0][i] != vstr[j][i])
						return vstr[0].substr(0, i);//下标为i时没有成功，所以截取长度不是i+1，而是为i
			return vstr[0];
		}
		//Valid Number
		//Valid Float:如果是Python:
		//import re ; re.match(pattern,string); pattern: r"[-+]?(\d+\.?|\.\d+)\d*([e|E][-+]?\d+)?"
		//Float要考虑到所有的合法测试用例:+123 .123 - .123 .123E10 123.E10 和非法用例.E - 10 .e
		//C++则如下
		//调用strtod(char* str,char** PtrEnd)
		//#include<cstdlib>
		bool isNumber(const string& s) {
			char* endptr;
			int c = 0;
			strtod(s.c_str(), &endptr);
			if (endptr == s.c_str()) return false;
			while (endptr) {
				if (!isspace(*endptr))
					return false;
				++endptr;
			}
			return true;
		}
		//Integer And RomanNumber InterTrasition
		//时间复杂度和Integer的大小有关，空间复杂度O(1)
		class IntegerEXRomanNum {
		public:
			//Integer to RomanNumber(1-3999)
			string IntegerToRoman(int n) {
				if (n < 1 || n>3999)
					return "";
				string ret;
				for (int i = 0; n>0; ++i) {
					int count = n / value[i];
					n %= value[i];
					for (int j = 0; j < count; ++j)
						ret += symbol[i];
				}
				return ret;
			}
			//RomanNumber to Integer(1-3999)IIX
			int RomanToInteger(const string& s) {
				if (s.length() == 0)
					return 0;
				int num = 0;
				for (int i = 0; i < s.length(); ++i) {
					if (i > 0 && s[i - 1] <= s [i])
						num += (map(s[i]) - 2 * map(s[i - 1]));
					else
						num += map(s[i]);
				}
				return num;
			}
		private:
			const int value[13] = { 1000,900,500,400,100,90,50,40,10,9,5,4,1 };//在类里面内置的类型无法使用模糊定义int value[]={1}
			const string symbol[13] = { "M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I" };
			int map(const char& c) {
				switch (c)
				{
				case 'M':return 1000;
				case 'D':return 500;
				case 'C':return 100;
				case 'L':return 50;
				case 'X':return 10;
				case 'V':return 5;
				case 'I':return 1;
				default:
					return 0;
				}
			}
		};
		//Count And Say
		//时间复杂度O(n2)，空间复杂度O(1)
		class CountAndSay {
		public:
			string NthCountAndSay(int n) {
				string s = "1";
				for(int i=1;i<n;++i)
					s = getNext(s);
				return s;
			}
		private:
			string getNext(const string& s) {
				string rets = "";
				for (auto i = s.begin(); i != s.end();) {
					auto j = find_if(i, s.end(),bind2nd(not_equal_to<char>(),*i));
					rets += (distance(i, j) + '0');
					rets += *i;
					i = j;
				}
				return rets;
			}
		};
		//Anagrams
		//时间复杂度O(n)，空间复杂度O(n)
		vector<string> Anagrams(vector<string>& v) {
			unordered_map<string, vector<string>> map;
			for (auto s: v) {
				string key = s;
				sort(key.begin(), key.end());
				map[key].push_back(s);
			}
			vector<string> vret;
			for (auto i = map.begin(); i != map.end(); ++i) {
				if (i->second.size() > 1)
					vret.insert(vret.begin(), i->second.begin(), i->second.end());
			}
			return vret;
		}
		//Simplify Path
		//时间复杂度O(Length of string)
		string SimplifyPath(string& s) {
			int i = 0;
			vector<string> v;
			while (i < s.size()) {
				while (i<s.size() && s[i] == '/') ++i;
				int start = i;
				if (start == s.size()) break;
				while (i<s.size() && s[i] != '/') ++i;
				string ss = s.substr(start, i - start);
				if (ss == "..") {
					if (!v.empty())
						v.pop_back();
				}
				else if (ss != ".")
					v.push_back(ss);
			}
			if (v.empty())
				return "/";
			string sret;
			for (auto i : v)
				sret += "/" + i;
			return sret;
		}
		//Length Of LastWord
		//string s consists of upper/lower-case alphabets and empty space characters ' '
		int LengthOfLastWord(string& s) {
			auto i = find_if(s.rbegin(), s.rend(), ::isalpha);
			if (i == s.rend())
				return 0;
			auto j = find_if_not(i, s.rend(), ::isalpha);
			return distance(i, j);
		}
	}
namespace StackAndSequence {
	//判断成对的括号是否有效
	bool IsValidParenthese(const string& s) {
		stack<char> stk;
		for (auto c : s) {
			if (c == '(' || c == '[' || c == '{')
				stk.push(c);
			else {
				if (stk.empty()) return false;
				if (c == ')'&& stk.top() != '(') return false;
				if (c == ']'&& stk.top() != '[') return false;
				if (c == '}'&& stk.top() != '{') return false;
				stk.pop();
			}
		}
		return stk.empty();
	}
	//最长的有效括号序列
	//Input:"))()([)][()[]])" Output:6
	//时间复杂度O(n)，空间复杂度O(n)
	int longestValidparenthese(const string& s) {
		if (s.empty()) return -1;
		stack<int> stk;
		int start;
		int maxLen = 0;
		for (int i = 0; i < s.size(); ++i) {
			if (s[i] == '(')
				stk.push(i);
			else {
				if (stk.empty()) start = i + 1;
				else {
					stk.pop();
					maxLen = stk.empty() ? max(maxLen, i - start + 1) : max(maxLen, i - stk.top());
				}
			}
		}
		return maxLen;
	}
	int LongestValidParenthese(const string& s) {
		stack<char> stk;
		stk.push(-1);
		int maxLen = 0;
		for (int i = 0; i < s.size(); ++i) {
			if (s[i] == '(' || s[i] == '[')
				stk.push(i);
			else {
				if (stk.top() == -1) {
					stk.push(i);
					continue;
				}
				if ((s[i] == ')'&&s[stk.top()] == '(')
					|| (s[i] == ']'&&s[stk.top()] == '[')) {
					stk.pop();
					maxLen = max(maxLen, i - stk.top());
				}
				else
					stk.push(i);

				if (stk.empty())
					stk.push(i);
			}
		}
		return maxLen;
	}
	//Largest Rectangle In Histogram
	//核心解法思路：构造递增的栈 https://www.cnblogs.com/ganganloveu/p/4148303.html#undefined
	//时间复杂度O(n)，空间复杂度O(n)
	int LargestRectangleInHistogram(vector<int>& v) {
		if (v.empty())
			return 0;
		v.push_back(0);
		stack<int> stk;
		int ret = 0;
		int i = 0;
		while( i < v.size()) {
			if (stk.empty() || v[i] > stk.top())
				stk.push(i++);
			else {
				int tmp = stk.top();
				stk.pop();
				ret = max(ret, v[tmp] * (stk.empty() ? i:i-tmp));
			}
		}
		return ret;
	}
	//Evaluate Reverse Polish Notation "求逆波兰表达式"
	class ReversePolishNotation {
	public:
		int EvaluateRPNotation(const string& s) {
			stack<int> stk;
			for (auto c : s) {
				if (c >= '0'&&c <= '9')
					stk.push(c - '0');
				else {
					int a = stk.top();
					stk.pop();
					int b = stk.top();
					stk.pop();
					int c = OperatorImpl(a, b, c);
					stk.push(c);
				}
			}
			return stk.top();
		}
	private:
		int OperatorImpl(int a, int b, char oper) {
			switch (oper)
			{
			case '+':return a + b;
			case '-':return a - b;
			case '*':return a*b;
			case '/':return a / b;
			case '%':return a%b;
			}
		}
	};
}
namespace Tree {
	struct TreeNode {
		int val;
		TreeNode* left = nullptr;
		TreeNode* right = nullptr;
		TreeNode(int value):val(value){}
	};
	//Implement Pre-Order Traversal For Binary Tree With Iterative Solution
	//时间复杂度O(n)，空间复杂度O(n)
	vector<int> PreOrder(TreeNode* root) {
		vector<int> ret;
		if (root == nullptr)
			return ret;
		stack<TreeNode*> stk;
		stk.push(root);
		while (!stk.empty())
		{
			TreeNode* tmp = stk.top();
			stk.pop();
			ret.push_back(tmp->val);
			if (tmp->right) stk.push(tmp->right);
			if (tmp->left) stk.push(tmp->left);

		}
		return ret;
	}
	//Implement In-Order Traversal For Binary Tree With Iterative Solution
	//时间复杂度O(n)，空间复杂度O(n)
	vector<int> InOrder(TreeNode* root) {
		vector<int> ret;
		if (root == nullptr)
			return ret;
		TreeNode* p = root;
		stack<TreeNode*> stk;
		while (!stk.empty() || p != nullptr) {
			if (p != nullptr) {
				stk.push(p);
				p = p->left;
			}
			else {
				TreeNode* tmp = stk.top();
				stk.pop();
				ret.push_back(tmp->val);
				p = tmp->right;
			}
		}
		return ret;
	}
	//Implement Post-Order Traversal For Binary Tree With Iterative Solution
	//时间复杂度O(n)，空间复杂度O(n)
	vector<int> PostOrder(TreeNode* root) {
		vector<int> ret;
		if (root == nullptr)
			return ret;
		stack<TreeNode*> stk;
		stk.push(root);
		while (!stk.empty())
		{
			TreeNode* tmp = stk.top();
			stk.pop();
			ret.push_back(tmp->val);

			if (tmp->left) stk.push(tmp->left);
			if (tmp->right) stk.push(tmp->right);
		}
		reverse(ret.begin(), ret.end());
		return ret;
	}
	//广度优先遍历二叉树 Level-Traversal
	//时间复杂度O(n),空间复杂度O(n)
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
				level.push_back(tmp->val);
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
	//ZigZag Level Order Traversal
	//用一个bool记录从左到右还是从右到左，每一层结束就reverse()一下
	//时间复杂度O(n)，空间复杂得多O(n)
	vector<vector<int>> ZigZagLevelOrderTraversal(TreeNode* root) {
		bool left_to_right = true;
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
				level.push_back(tmp->val);
				
				if (tmp->left) next.push(tmp->left);
				if (tmp->right) next.push(tmp->right);
			}
			if (!left_to_right) reverse(level.begin(), level.end());
			left_to_right = !left_to_right;//下一行的取法顺序
			ret.push_back(level);
			swap(current, next);
		}
		return ret;
	}
	//Is Same Tree
	//时间复杂度O(lgn)，递归版空间复杂度O(1)，迭代版空间复杂度O(n)
	bool IsSameTree(TreeNode* p,TreeNode* q) {
		if (!p && !q) return true;
		if (!p || !q) return false;
		return p->val == q->val&&IsSameTree(p->left, q->left) && IsSameTree(p->right, q->right);
	}
	//迭代版
	bool IsSameTreeI(TreeNode* p, TreeNode* q) {
		if (!p && !q) return true;
		if (!p || !q) return false;
		queue<TreeNode*> quee;
		quee.push(p);
		quee.push(q);
		while (!quee.empty())
		{
			TreeNode* fst = quee.front();
			quee.pop();
			TreeNode* snd = quee.front();
			quee.pop();
			if (!fst && !snd) continue;
			if (!fst || !snd) return false;
			if (fst->val != snd->val) return false;
			quee.push(p->left);
			quee.push(q->left);
			quee.push(p->right);
			quee.push(q->right);
		}
		return true;
	}
	class SymTree {
	public:
		//Is Symmetric Tree （镜像树）
		//递归版
		bool IsSymmetricTreeI(TreeNode* root) {
			if (root == nullptr)
				return true;
			return IsSymmetricTree(root->left, root->right);
		}
		bool IsSymmetricTree(TreeNode* p, TreeNode* q) {
			if (!p && !q) return true;
			if (!p || !q) return false;
			return p->val == q->val&&IsSymmetricTree(p->left, q->right) && IsSymmetricTree(p->right, q->left);
		}
		//迭代版
		bool IsSymmetricTree(TreeNode* root) {
			if (root == nullptr)
				return true;
			queue<TreeNode*> quee;
			quee.push(root->left);
			quee.push(root->right);
			while (!quee.empty()) {
				auto p = quee.front();
				quee.pop();
				auto q = quee.front();
				quee.pop();

				if (!p && !q) continue;
				if (!p || !q) return false;
				if (p->val != q->val) return false;

				quee.push(p->left);
				quee.push(q->right);

				quee.push(p->right);
				quee.push(q->left);
			}
			return true;
		}
	};
	class TreeHeight {
		//Is Balanced Binary Search Tree
		//递归求二叉树高度时加入平衡判断
		//时间复杂度O(logn)，空间复杂度O(1)
		bool IsBalancedTreeI(TreeNode* root) {
			if (root == nullptr)
				return true;
			return HeightAndBalance(root) >= 0;

		}
		int HeightAndBalance(TreeNode* root) {
			if (root == nullptr)
				return 0;
			int lHeight = HeightAndBalance(root->left);
			int rHeight = HeightAndBalance(root->right);

			if (lHeight < 0 || rHeight < 0 || abs(lHeight - rHeight)>1) return -1;//判断是否为平衡二叉树

			return max(lHeight, rHeight) + 1;//三方合并计算树的高
		}
		//还可以通过广度优先遍历时累计高度
	};

	//Flatten Binary Tree To Linked List
	// https://www.cnblogs.com/grandyang/p/4293853.html 解法二
	//迭代法
	void FlattenTree(TreeNode* root) {
		if (root == nullptr)
			return;
		TreeNode* cur = root;
		while (cur)
		{
			if (cur->left) {
				TreeNode* pr = cur->left;
				while (pr->right) pr = pr->right;
				pr->right = cur->right;
				cur->right = cur->left;
				cur->left = nullptr;
			}
			cur = cur->right;
		}
	}
	//Populating Next Right Pointers Each Node II
	//时间复杂度O(n)，空间复杂度O(1)
	class PopulatingNextRightPointers {
		struct TreeLinkNode {
			TreeLinkNode* left = nullptr;
			TreeLinkNode* right = nullptr;
			TreeLinkNode* next = nullptr;
			int val;
			TreeLinkNode(int value):val(value){}
		};
		void PopulatingEachNode(TreeLinkNode* root) {
			if (root == nullptr) return;
			while (root)
			{
				TreeLinkNode* prev = nullptr;
				TreeLinkNode* next = nullptr;
				next = root->left ? root->left : root->right;
				while(root) {
					if (root->left) {
						if (prev) prev->next = root->left;
						prev = root->left;
					 }
					if (root->right) {
						if (prev) prev->next = root->right;
						prev = root->right;
					}
					root = root->next;
				}
				root = next;//转到下一个Level
			}
		}
	};
	class BuildTreeI {
		//Building Tree From Pre-order and In-order
	public:
		TreeNode* BuildTree(vector<int>& pre, vector<int>& in) {
			return BuildTree(pre, pre.begin(), pre.end(), in, in.begin(), in.end());
		}
	private:
		TreeNode* BuildTree(const vector<int>& pre, vector<int>::iterator pre_left, vector<int>::iterator pre_right,
			const vector<int>& in, vector<int>::iterator in_left, vector<int>::iterator in_right) {
			if (pre_left == pre_right || in_left == in_right) return nullptr;
			TreeNode* cur = new TreeNode(*pre_left);
			auto rt = find(in_left, in_right, *pre_left);
			int len = distance(in_left, rt);
			cur->left = BuildTree(pre, pre_left + 1, pre_left + len+1, in, in_left, rt);
			cur->right = BuildTree(pre, pre_left + len + 1, pre_right, in, next(rt), in_right);
			return cur;
		}
	};
	class BuildTreeII {
		//Building Tree From Post-order and In-order
	public:
		TreeNode* BuildTree(vector<int>& post, vector<int>& in) {
			return BuildTree(post, post.begin(), post.end(), in, in.begin(), in.end());
		}
	private:
		TreeNode* BuildTree(const vector<int>& post, vector<int>::iterator post_left, vector<int>::iterator post_right,
			const vector<int>& in, vector<int>::iterator in_left, vector<int>::iterator in_right) {
			if (post_left == post_right || in_left == in_right) return nullptr;
			TreeNode* cur = new TreeNode(*post_right);
			auto rt = find(in_left, in_right, *post_left);
			int len = distance(in_left, rt);
			cur->left = BuildTree(post, post_left, post_left + len, in, in_left, rt);
			cur->right = BuildTree(post, post_left + len, prev(post_right), in, next(rt), in_right);
			return cur;
		}
	};
	class UniqueBSTs {
	public:
		//给定N个节点值，返回能构造成二叉搜索树的数量
		//Solution:DP
		int UniqueBST(int n) {
			vector<int> vf;
			vf[0] = 0;
			vf[1] = 1;
			int i;
			for (i = 2; i <= n; ++i)
				for (int k = 1; k <= i; ++k)
					vf[i] = vf[k - 1] * vf[i - k];
			return vf[n];
		}
	};
	//Is a Valid BST ?
	bool isValidBST(TreeNode* root) {
		if (root == nullptr) return true;
		if (root->left && root->left->val > root->val) return false;
		if (root->right && root->right->val < root->val) return false;
		return isValidBST(root->left) && isValidBST(root->right);
	}
	//Transform SortedArray to Balanced BST
	//二分法
	class transformSortedArrayToBST {
	public:
		TreeNode* TransSortedArrayToBST(const vector<int>& v) {
			if (v.size() == 0) return nullptr;
			return TransSArrayToBST(v,0,v.size()-1);
		}
	private:
		TreeNode* TransSArrayToBST(const vector<int>& vs,int first,int last) {
			if (first > last) return nullptr;
			int mid = first + (last - first) / 2;
			TreeNode* cur = new TreeNode(vs[mid]);
			cur->left = TransSArrayToBST(vs, first, mid - 1);
			cur->right = TransSArrayToBST(vs, mid + 1, last);
			return cur;
		}
	};
	/////////////////
	struct ListNode {
		int val;
		ListNode* next = nullptr;
		ListNode(int value):val(value){}
	};
/////////////////////////////////////////////
//Transform Sorted LinkedList to Binary Search Tree
//二分法，快慢指针
	class TransSLinkListToBST {
	public:
		TreeNode* TransLinkListToBST(ListNode* head) {
			return TransLListToBST(head);
		}
	private:
		TreeNode* TransLListToBST(ListNode* head) {
			if (!head) return nullptr;
			if (!head->next) return new TreeNode(head->val);
			ListNode* slow = head;
			ListNode* fast = head;
			ListNode* prev;
			while (fast && fast->next)
			{
				prev = slow;
				slow = slow->next;
				fast = fast->next->next;
			}
			ListNode* next = slow->next;
			prev->next = nullptr;
			TreeNode* root = new TreeNode(slow->val);
			root->left = TransLListToBST(head);
			root->right = TransLListToBST(next);
			return root;
		}
	};
	//Minium Deepth Of BT
	int minDepth(TreeNode* root) {
		if (root == nullptr) return 0;
		if (root->left == nullptr) return 1 + minDepth(root->right);
		if (root->right == nullptr) return 1 + minDepth(root->left);
		return 1 + min(minDepth(root->left), minDepth(root->right));
	}
	//广搜
	int minDepthIter(TreeNode* root) {
		if (root == nullptr) return 0;
		int dep = 1;
		queue<TreeNode*> next;
		queue<TreeNode*> cur;
		cur.push(root);
		while (!cur.empty())
		{
			while (!cur.empty())
			{
				TreeNode* tnode = cur.front();
				cur.pop();
				if (!tnode->left && !tnode->right) return dep;
				if (tnode->left) next.push(tnode->left);
				if (tnode->right) next.push(tnode->right);
			}
			dep++;
			swap(cur, next);
		}
	}
	//Maxium Deepth Of BT
	int maxDepth(TreeNode* root) {
		if (root == nullptr) return 0;
		return 1 + max(maxDepth(root->left), maxDepth(root->right));
	}
	//Is exit a Path from root to leaf which Sum is a value given
	bool sumOfPath(TreeNode* root,int sum) {
		if (root == nullptr) return false;
		if (root->left == nullptr&&root->right == nullptr)
			return sum == root->val;
		return sumOfPath(root->left, sum - root->val) || sumOfPath(root->right, sum - root->val);
	}
	//深搜
	class FindSumPath {
	public:
		//Find all of Paths whose sum is equal to a value given
		vector<vector<int>> findAllsumOfPath(TreeNode* root, int sum) {
			vector<vector<int>> vret;
			vector<int> vt;
			FindPath(root, sum, vret, vt);
			return vret;
		}
	private:
		void FindPath(TreeNode* cur, int sum, vector<vector<int>>& vrt, vector<int>& vt) {
			if (cur == nullptr) return;
			vt.push_back(cur->val);
			if (cur->left == nullptr&&cur->right == nullptr)
				if (sum == cur->val)
					vrt.push_back(vt);
			FindPath(cur->left, sum - cur->val, vrt, vt);
			FindPath(cur->right, sum - cur->val, vrt, vt);
			vt.pop_back();
		}
	};
	//法二：迭代，中序遍历
	vector<vector<int>> findAllsumOfPath(TreeNode* root, int sum) {
		vector<vector<int>> vret;
		if (root == nullptr) return vret;
		stack<TreeNode*> stk;
		vector<int> vt;
		TreeNode* cur = root;
		while (!stk.empty() || cur)
		{
			if (cur) {
				stk.push(cur);

				vt.push_back(cur->val);
				if (cur->left==nullptr && cur->right==nullptr && sum == cur->val)
					vret.push_back(vt);
				sum -= cur->val;

				cur = cur->left;
			}else{
				cur = stk.top();
				stk.pop();
				if (cur->left) {   //左节点存在才sum回溯
					vt.pop_back();
					sum += cur->val;
				}
				cur = cur->right;
			}
		}
		return vret;
	}
	//Maxium Sum Path,Return Maxium Vaule
	//类似于Maxium Substring Sum，分支和大于0则累加，小于零则丢弃
	class MaxiumSumPath {
	public:
		int maxSumPath(TreeNode* root) {
			int iret = INT_MIN;
			maxSumPath(root, iret);
			return iret;
		}
	private:
		int maxSumPath(TreeNode* cur, int& res) {
			if (cur == nullptr) return 0;
			int left = max(maxSumPath(cur->left, res), 0);
			int right= max(maxSumPath(cur->right, res), 0);
			res = max(res, left + right + cur->val);
			return max(left, right) + cur->val;
		}
	};
	//Populating Next Right Pointer Each Nodes
	struct TreeLinkNode {
		TreeLinkNode* left = nullptr;
		TreeLinkNode* right = nullptr;
		TreeLinkNode* next = nullptr;
		int val;
		TreeLinkNode(int value) :val(value) {}
	};
	void PopulatingAllRightPointers(TreeLinkNode* root) {
		if (root == nullptr) return;
		TreeLinkNode* cur = root;
		TreeLinkNode* pre = nullptr, *next = root;
		while (next)
		{
			for (; cur; cur = cur->next) {
				next = cur->left ? cur->left : cur->right;
				if (cur->left) {
					if (pre) pre->next = cur->left;
					pre = cur->left;
				}
				if (cur->right) {
					if (pre) pre->next = cur->right;
					pre = cur->right;
				}
			}
			pre = next;
		}
	}
	//Sum Root To Leaf Numbers
	//1->2->3 组成了数字123，类似的，将所有根到叶子所组成的数字相加,数字由0-9构成
	class SumRootToLeafNumbers {
	public:
		int Sum(TreeNode* root) {
			if (root == nullptr) return 0;
			vector<int> vt;
			SumRootToLeafNumber(root, vt);
			return sum;
		}
	private:
		int sum = 0;
		void SumRootToLeafNumber(TreeNode* cur, vector<int>& vt) {
			if (cur == nullptr) return;
			vt.push_back(cur->val);
			if (cur->left == nullptr&&cur->right == nullptr)
				sum += sumOfvector(vt);
			SumRootToLeafNumber(cur->left, vt);
			SumRootToLeafNumber(cur->right, vt);
			vt.pop_back();
		}
		int sumOfvector(vector<int>& vt) {
			int sum = 0;
			for (auto i : vt)
				sum = 10 * sum + i;
			return sum;
		}
	};
}
namespace sorting {
	//Merge Two Sorted Arrays
	void mergeTwoSortedArrays(vector<int>& a, vector<int>& b) {
		int ai = a.size() - 1;
		int bi = b.size() - 1;
		int mi = a.size() + b.size() - 1;
		while (ai >= 0 && bi >= 0)
			a[mi--] = a[ai] > b[bi] ? a[ai--] : b[bi--];
		while (bi >= 0)
			a[mi--] = b[bi--];
	}
	//Merge Two Sorted Linked Lists
	struct ListNode
	{
		int val;
		ListNode* next = nullptr;
		ListNode(int value):val(value){}
	};
	ListNode* MergeTwoSortedLinkedList(ListNode* l1, ListNode* l2) {
		if (l1 == nullptr) return l2;
		if (l2 == nullptr) return l1;
		ListNode dummy(-1);
		ListNode* prev = &dummy;
		while (l1&&l2) {
			if (l1->val > l2->val) {
				prev->next = l2; l2 = l2->next;
			}
			else {
				prev->next = l1; l1 = l1->next;
			}
			prev = prev->next;
		}
		prev->next = l1 == nullptr ? l2 : l1;
		return dummy.next;
	}
	//归并K个链表
	class mergeKSortedLinkedList {
	public:
		ListNode* mergeKList(vector<ListNode*>& vl) {
			ListNode* cur = vl[0];
			for (int i = 1; i < vl.size(); ++i)
				cur = mergeTwoList(cur, vl[i]);
			return cur;
		}
	private:
		ListNode* mergeTwoList(ListNode* l1, ListNode* l2) {
			if (l1 == nullptr) return l2;
			if (l2 == nullptr) return l1;
			ListNode dummy(-1);
			ListNode* prev = &dummy;
			while (l1&&l2)
			{
				if (l1->val > l2->val) {
					prev->next = l2; l2 = l2->next;
				}
				else {
					prev->next = l1; l1 = l1->next;
				}
				prev = prev->next;
			}
			prev->next = l1 == nullptr ? l2 : l1;
			return dummy.next;
		}
	};
	//Insertion Sort List 插排
	//依次取出node，从头扫描链表，把它放到 dummy(-1)->null合适的位置
	//时间复杂度O(n**2)，空间复杂度O(1)
	class InsertionSortedList{
	public:
		ListNode* InsertLinkedList(ListNode* head) {
			if (head == nullptr) return nullptr;
			ListNode dummy(-1);
			for (ListNode* cur = head; cur;) {
				ListNode* next = cur->next;
				ListNode* pos = findPosition(&dummy, cur->val);
				cur->next = pos->next;
				pos->next = cur;
				cur = next;
			}
			return dummy.next;
		}
	private:
		ListNode* findPosition(ListNode* head, int val) {
			ListNode* prev = nullptr;
			ListNode* cur = head;
			while (cur&&cur->val <= val) {
				prev = cur;
				cur = cur->next;
			}
			return prev;
		}
	};
	//Sort Linked List,requirements:time:O(N*logn),space:const
	//通常单链表用归并，双链表用快排
	class LinkedListSorted {
		ListNode* mergeSorted(ListNode* head) {
			if (head == nullptr || head->next == nullptr) return head;
			//快慢指针来找中点
			ListNode* fast = head, *slow = head, *prev = nullptr;

			while (fast&&fast->next) {
				prev = slow;
				slow = slow->next;
				fast = fast->next->next;
			}
			prev->next = nullptr;
			ListNode* left = mergeSorted(head);
			ListNode* right = mergeSorted(slow);
			return mergeTwoList(left, right);
		}
	private:
		ListNode* mergeTwoList(ListNode* l1, ListNode* l2) {
			if (l1 == nullptr) return l2;
			if (l2 == nullptr) return l1;
			ListNode dummy(-1);
			ListNode* prev = &dummy;
			while (l1&&l2)
			{
				if (l1->val < l2->val) {
					prev->next = l1;
					l1 = l1->next;
				}
				else {
					prev->next = l2;
					l2 = l2->next;
				}
			}
			prev->next = l1 == nullptr ? l2 : l1;
			return dummy.next;
		}
	};
	//First Missing Positive
	//找到第一个缺失的正数,桶排
	//时间复杂度O(n)，空间复杂度O(1)
	class FirstMissingPosivite {
	public:
		int FirstMissingPlus(vector<int>& v) {
			bucketSort(v);
			for (int i = 0; i < v.size(); ++i)
				if (v[i] != i + 1)
					return i + 1;
			return v.size() + 1;
		}
	private:
		void bucketSort(vector<int>& vbkt) {
			for (int i = 0; i < vbkt.size(); ++i) {
				while (vbkt[i] != i + 1) {
					if (vbkt[i] <= 0 || vbkt[i] > vbkt.size() || vbkt[vbkt[i] - 1] == vbkt[i])//小于等于零和大于边界的值等着被其它属于该位置的数swap
						break;
					swap(vbkt[i], vbkt[vbkt[i] - 1]);
				}
			}
		}
	};
	//Sort Colors
	//We use the integers 0,1,2 to represent the color red, white,blue.
	//Sort them so thatthe same color are adjacent, with the colors in the order red, white and blue
	//Simple Solution:Two-Pass
	void TwoPass(vector<int>& v) {
		int color[3] = { 0 };
		for (int i = 0; i < v.size(); ++i)
			color[v[i]]++;
		int cur = 0;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < color[i]; ++j)
				v[cur++] = i;
	}
	//Advanced Solution:One-Pass，只扫描一遍
	void OnePass(vector<int>& v) {
		int red = 0, blue = v.size() - 1;
		for (int i = 0; i <= blue; ++i) {
			if (v[i] == 0)
				swap(v[i++], v[red++]);
			else if (v[i] == 2)
				swap(v[i++], v[blue--]);
			else
				++i;
		}
	}
	//More Effective Solution:STL's partition()
	void Partition(vector<int>& v) {
		partition(partition(v.begin(), v.end(), bind1st(equal_to<int>(), 0)), v.end(), bind1st(equal_to<int>(), 1));
	}
}
namespace Searching {
	//对有序的搜索，毫无疑问用二分法
	//Search A Range For Sorted Array
	//Requirement:Time with O(logN)
	vector<int> SearchForRange(vector<int>& v,int target) {
		sort(v.begin(), v.end());
		auto range = equal_range(v.begin(), v.end(),target);
		if (range.first == range.second)
			return  vector<int>{-1, -1};
		return vector<int>{range.first - v.begin(), range.second - v.begin() - 1};
	}
	//Search Insert Position
	//已排序的Array中，target存在则返回它的位置，否则返回该target应该所处的位置,With Requirement:Time Complex withIn O(logN)
	//直接用bound_lower或再实现bound_lower
	class ReImpleBoundLower {
	public:
		int SearchForPosition(const vector<int>& v,int target) {
			auto pos = lower_bound(v.begin(), v.end(), target);
			return pos - v.begin();
		}
	private:
		template<typename Iter, typename T>
		Iter lower_bound(Iter begin, Iter end, T value) {
			while (begin < end) {
				Iter mid = next(begin, distance(begin, end) / 2);
				if (*mid < value) begin = ++mid;
				else
					end = mid;
			}
			return end;
		}
		template<typename Iter, typename T>
		Iter upper_bound(Iter begin, Iter end, T value) {
			while (begin < end) {
				Iter mid = next(begin, distance(begin, end) / 2);
				if (*mid <= value) begin = ++mid;
				else
					end = mid;
			}
			return end;
		}
	};
	//Search A 2D Matrix
	//时间复杂度O(logN)
	//For example, Consider the following matrix:
	//[
	//	[1, 3, 5, 7],
	//	[10, 11, 16, 20],
	//	[23, 30, 34, 50]
	//]
	//Given target = 3, return true.
	bool searchIn2DMatrix(vector<vector<int>>& vv,int target) {
		if (vv.empty()) return false;
		int m = vv.size();
		int n = vv[0].size();
		int i;
		for (i = 0; i < m; ++i)
			if (vv[i][n - 1] > target)
				break;
		if (i == m) return false;
		//Next Binary Search
		int first = 0, second = n - 1;
		while (first<=second)
		{
			int mid = (second - first) / 2 + first;
			if (vv[i][mid] == target) return true;
			else if (vv[i][mid] < target) first = ++mid;
			else
				second = --mid;
		}
		return false;
	}
}
namespace ForceEnum {
	//Subset,全集没有重复
	//迭代法
	//选或不选->时间复杂度O(2**N)，空间复杂度O(1)
	vector<vector<int>> Subset(vector<int>& v) {
		vector<vector<int>> vret(1);
		sort(v.begin(), v.end());
		for (auto i : v) 
			for (int j = 0; j < vret.size(); ++j) {
				vret.push_back(vret[j]);
				vret.back().push_back(i);
			}
		return vret;
	}
	//深搜，但是没有剪枝，称之递归更妥
	class subSetRecursion {
	public:
		vector<vector<int>> subSet(vector<int>& v) {
			sort(v.begin(), v.end());
			vector<vector<int>> vret;
			vector<int> cur;
			Recursion(v, cur, vret, 0);
			return vret;
		}
	private:
		void Recursion(const vector<int>&v, vector<int>&cur, vector<vector<int>>& vret, int start) {
			vret.push_back(cur);
			for (int i = start; i < v.size(); ++i) {
				cur.push_back(v[i]);
				Recursion(v, cur, vret, i + 1);
				cur.pop_back();
			}
		}
	};
	//More More Effective 二进制法
	//前提是元素总量不超过int的位数,{A,B,C,D}中6==0110表示子集{B,C}
	vector<vector<int>> subsetWithBinary(vector<int>& v) {
		sort(v.begin(), v.end());
		vector<vector<int>> vret;
		vector<int> cur;
		size_t n = v.size();
		for (size_t i = 0; i < 2 << n; ++i) {
			for (size_t j = 0; j < n; ++j)
				if (i & 1 << j) cur.push_back(v[j]);
			vret.push_back(cur);
			cur.clear();
		}
		return vret;
	}
	//若全集元素有重复，如求{1,2,2}的子集,则在上面的基础上push子集之前先判重，或借助STL的数据结构set()天然去重
	vector<vector<int>> subSetWithBinary(vector<int>& v) {
		sort(v.begin(), v.end());
		vector<vector<int>> vret;
		vector<int> cur;
		unordered_set<string> set; //这种写法需要vector<>模板支持hash()，参见https://blog.csdn.net/haluoluo211/article/details/82468061
		int n = v.size();
		for (int i = 0; i < 2 << n; ++i) {
			string s;
			for (int j = 0; j < n; ++j)
				if (i & 1 << j) {
					s += to_string(v[j]);
					cur.push_back(v[j]);
				}
			if (!count(set.begin(), set.end(), s)) {
				set.insert(s);
				vret.emplace_back(cur);
			}
			cur.clear();
		}
		return vret;
	}
	//Permutation
	//OJ网站代码测试
	//Solution I: Call std::next_premutation(num.begin(),num.end())
	vector<vector<int>> Permutation(vector<int>& vnum) {
		vector<vector<int>> vret;
		if (vnum.empty()) return vret;
		sort(vnum.begin(), vnum.end());
		do {
			vret.push_back(vnum);
		} while (next_permutation(vnum.begin(), vnum.end()));
		return vret;
	}
	//面试手撕算法
	//Solution II:Re-Implement next_permutation()
	class Permutations {
	public:
		vector<vector<int>> getPermutation(vector<int>& vnum) {
			vector<vector<int>> vret;
			if (vnum.empty()) return vret;
			sort(vnum.begin(), vnum.end());
			do {
				vret.push_back(vnum);
			} while (next_permutation(vnum.begin(), vnum.end()));
			return vret;
		}
	private:
		template<typename Iter>
		bool next_permutation(Iter begin, Iter end) {
			auto rbegin = reverse_iterator<Iter>(end);
			auto rend = reverse_iterator<Iter>(begin);
			auto cur = next(rbegin);
			while (cur != rend&& *cur > *prev(cur))
				++cur;
			if (cur == rend) {
				reverse(begin, end);
				return false;
			}
			auto greatPos = find_if(rbegin, cur, bind1st(less<int>(), *cur));
			//auto greatPos = upper_bound(rbegin, cur, *cur);
			swap(*cur, *greatPos);
			reverse(rbegin, cur);
			return true;

		}
	};
	//康拓编解码
	//判断这个数是第几个排列，康拓编码
	int getKth(vector<int>& vnum) {
		int n = vnum.size();
		int* f = new int[n];
		f[0] = 1;
		for (int i = 1; i < n; ++i)
			f[i] = i*f[i - 1];
		int k = 0;
		for (int i = 0; i < n;++i) {
			int amt = count_if(next(vnum.begin(), i), vnum.end(),bind1st(greater<int>(),vnum[i]));
			k += amt*f[n - 1 - i];
		}
		delete[]f;
		return k + 1;//有k个比它小的排列，所以它是第k+1个
	}
	////找到第k个排列
	//排序后调用k-1次next_permutation()
	//Advaced:康拓解码
	vector<int> findKthPermutation(vector<int>& vnum,int k) {
		vector<int> vret;
		if (vnum.empty()) return vret;
		if (k <= 0) return vret;
		sort(vnum.begin(), vnum.end());
		int n = vnum.size();
		int* f = new int[n];
		f[0] = 1;
		for (int i = 1; i < n; ++i)
			f[i] = i*f[i - 1];
		--k;
		while (k!=0)
		{
			auto less = next(vnum.begin(),k / f[--n]);
			vret.push_back(*less);
			vnum.erase(less);
			k = k%f[n];
		}
		vret.push_back(vnum[0]);
		delete[]f;
		return vret;
	}
	//如果给的排列中的元素允许重复，解法和不重复的一样
	//Combinations
	//Given two integers n and k, return all possible combinations of k numbers out of 1 ...n.
	//For example, If n = 4 and k = 2, a solution is :
	//[[2, 4],[3, 4],[2, 3],[1, 2],[1, 3],[1, 4],]
	//Solution I:Recursion
	//和求子集的递归很相似，这种求满足条件的总集的直觉是DFS
	class Combinations {
	public:
		vector<vector<int>> KCombinations(int n, int k) {
			vector<vector<int>> vret;
			if (n == 0 || k == 0 || k > n) return vret;
			vector<int> cur;
			dfs(n, k, 1, cur, vret);
			return vret;
		}
	private:
		void dfs(int n, int k, int start, vector<int>& cur, vector<vector<int>>& vret) {
			if (cur.size() > k) return;
			if (cur.size() == k) {
				vret.push_back(cur);
				return;
			}
			for (int i = start; i <= n; ++i) {
				cur.push_back(i);
				dfs(n, k, i + 1, cur, vret);
				cur.pop_back();
			}
		}
	};
	//Solution II: Iteration
	//1,2,3 (n=3,k=2)
	//[]
	//[],[1]
	//[],[2],[1] [1,2]->提取
	//[][3][2],[1] [1,3]->提取,[2,3]->提取
	//[1,2][1,3][2,3]
	//和求子集的迭代类似
	vector<vector<int>> combination(int n, int k) {
		vector<vector<int>> subset(1);
		vector<vector<int>> vret;
		for (int i = 1; i <= n; ++i) {
			for (int j = 0; j < subset.size(); ++j) {
				subset.push_back(vret[j]);
				subset.back().push_back(i);
				if (subset.back().size() == k) {
					vret.push_back(subset.back());
					subset.pop_back();
				}
			}
		}
		return vret;
	}
	//All Letter Combination
	//Recursion
	class LetterCombination {
	public:
		vector<string> getLetterCombination(const string& digit) {
			vector<string> vret;
			if (digit.empty()) return vret;
			string cur = "";
			doCombination(digit, 0, cur, vret);
			return vret;
		}
	private:
		const vector<string> vmap{ " ","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz" };
		void doCombination(const string& digit, int pos, string cur, vector<string>& vret) {
			if (pos == digit.size()) {
				vret.push_back(cur);
				return;
			}
			for (int i = 0; i < vmap[digit[pos] - '0'].size(); ++i)
				doCombination(digit, pos + 1, cur + vmap[digit[pos] - '0'][i], vret);
		}
		//第二种深搜写法，第一种是基于栈的临时变量cur的写法，第二种是变量引用&cur
		/*void doCombination(const string& digit, int pos, string& cur, vector<string>& vret) {
			if (pos == digit.size()) {
				vret.push_back(cur);
				return;
			}
			for (int i = 0; i < vmap[digit[pos] - '0'].size(); ++i) {
				cur.push_back(vmap[digit[pos] - '0'][i]);
				doCombination(digit, pos + 1, cur, vret);
				cur.pop_back();

			}
		}*/
	};
	//Iteration
	vector<string> phoneLetterCombination(const string& digit) {
		if (digit.empty()) return {};
		vector<string> vret{ "" };
		const vector<string> vmap{ " ","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz" };
		vector<string> cur;
		for (auto c : digit) {
			string stmp = vmap[c - '0'];
			for (auto i : stmp)
				for (int j = 0; j < vret.size();++j)
					cur.push_back(vret[j]+i);
			vret = cur;
			cur.clear();
		}
		return vret;
	}
}
namespace BFS {
	//广搜主要用来解决最短路径问题
	//他比深搜有更好的效率，但空间开销也更大
	//Word Ladder 
	//For example, Given:start = "hit";end = "cog";dict = ["hot", "dot", "dog", "lot", "log"]
	//As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog", return its length 5
	int wordLadder(const string& start, const string& end, vector<string>& dict) {
		unordered_set<string> word(dict.begin(),dict.end());
		queue<string> que;
		que.push(start);
		int layer = 0;
		while (!que.empty())
		{
			for (int i = 0; i < que.size(); ++i) {
				string cur = que.front();
				que.pop();
				for (int j = 0; j < cur.size(); ++j) {
					string stmp = cur;
					for (char c = 'a'; c <= 'z'; ++c) {
						stmp[j] = c;
						if (stmp == end) return layer + 1;
						if (find(word.begin(), word.end(), stmp) != word.end()) {
							que.push(stmp);
							word.erase(stmp);
						}
					}
				}
			}
			++layer;
		}
		return 0;
	}
	//Word Ladder II
	//返回所有最短路径的集合
	vector<vector<string>> wordLadderII(const string& start, const string& end, const vector<string>& dict) {
		unordered_set<string> dicts(dict.begin(), dict.end());
		queue<vector<string>> paths;
		vector<string> path = { start };
		paths.push(path);
		vector<vector<string>> vret;
		int minLadder = INT_MAX;
		while (!paths.empty())
		{
			vector<string> cur = paths.front();
			paths.pop();
			if (cur.size() > minLadder) //剪枝
				break;
			string word = cur.back();
			for (int i = 0; i < word.size(); ++i) {
				string newWord = word;
				for (char c = 'a'; c <= 'z'; ++c) {
					newWord[i] = c;
					vector<string> vtmp = cur;
					if (newWord == end) {
						vtmp.push_back(newWord);
						vret.push_back(vtmp);
						minLadder = cur.size();
						break;
					}
					if (find(dicts.begin(), dicts.end(), newWord) != dicts.end()) {
						vtmp.push_back(newWord);
						dicts.erase(newWord);
						paths.push(vtmp);
					}
				}
			}
		}
		return vret;
	}
	//求迷宫最短路径
	int mazeShortestPath(vector<vector<int>>& vv, int startx, int starty, int endx, int endy) {
		if (vv.empty()) return -1;
		int m = vv.size();
		int n = vv[0].size();
		int** visit = new int*[m];//可以用vector<vector<int>> vv(m,vector<int>(n,0))替代																					
		for (int i = 0; i < m; ++i) {
			visit[i] = new int[n];
			memset(visit[i], 0, n);
		}
		int dx[4] = { 0,0,-1,1 }, dy[4] = { -1,1,0,0 };
		typedef pair<int, int> P;
		queue<P> que;
		que.push(P(startx, starty));
		visit[startx][starty] = 1;
		queue<P> next;
		int layer = 0;
		while (!que.empty())
		{
			while (!que.empty())
			{
				P cur = que.front();
				que.pop();
				for (int i = 0; i < 4; ++i) {
					int curx = cur.first + dx[i];
					int cury = cur.second + dy[i];
					if (curx < 0 || curx >= m || cury < 0 || cury >= n || vv[curx][cury] || visit[curx][cury]) continue;
					visit[curx][cury] = 1;
					if (curx == endx&&cury == endy) {
						delete[]visit;
						return layer + 1;
					}
					next.push(P(curx, cury));
				}
			}
			layer++;
			swap(next, que);
		}
		delete[]visit;
		return layer;
	}
}
namespace DFS {
	//Surround Regions
	//类似于区域划分问题,DFS
	//需求：把被X包围的O用X填充
	//  X X X X      X X X X
	//  X O O X  ->  X X X X
	//	X X O X      X X X X
	//	X O X X      X O X X
	class SurRRegions {
	public:
		void surroundRegins(vector<vector<char>>& vv) {
			if (vv.empty()) return;
			int m = vv.size();
			int n = vv[0].size();
			for(int i=0;i<m;++i)
				for (int j = 0; j < n; ++j)
					if (vv[i][j] == 'O' && (i == 0 || i == m - 1 || j == 0 || j == n - 1))
						fillChar(vv, i, j,m,n);
			for (int i = 0; i<m; ++i)
				for (int j = 0; j < n; ++j) {
					if (vv[i][j] == 'O') vv[i][j] = 'X';
					else if (vv[i][j] == '#') vv[i][j] = 'O';
					else;
				}
		}
	private:
		void fillChar(vector<vector<char>>& vv, int i, int j,int m,int n) {
			if (i < 0 || i >= m || j < 0 || j >= n - 1 || vv[i][j] == 'X' || vv[i][j]=='#') return;
			vv[i][j] = '#';
			fillChar(vv, i - 1, j, m, n);
			fillChar(vv, i + 1, j, m, n);
			fillChar(vv, i, j - 1, m, n);
			fillChar(vv, i, j + 1, m, n);
		}
	};
	//Word Search
	//深搜特征比较突出的深搜命题
	class WordSearch {
	public:
		bool wordSearch(const vector<string>& dict, const string& s) {
			if (s.empty()||dict.empty()||dict[0].empty()) return false;
			int m = dict.size();
			int n = dict[0].size();
			vector<vector<bool>> visit(m, vector<bool>(n, false));
			for (int i = 0; i < m; ++i)
				for (int j = 0; j < n; ++j)
					if (dfs(dict, i, j, s, 0,visit))
						return true;
			return false;
		}
	private:
		bool dfs(const vector<string>& dict, int i, int j, const string& s, int pos, vector<vector<bool>>& visit) {
			if (pos == s.size()) return true;
			if (i < 0 || i >= dict.size() || j < 0 || j >= dict[0].size() || visit[i][j] || dict[i][j] != s[pos]) return false;
			visit[i][j] = true;
			if (dfs(dict, i - 1, j, s, pos + 1, visit) ||
				dfs(dict, i + 1, j, s, pos + 1, visit) ||
				dfs(dict, i, j - 1, s, pos + 1, visit) ||
				dfs(dict, i, j + 1, s, pos + 1, visit))
				return true;
			visit[i][j] = false;
			return false;
		}
	};
	//Paradrome Partition
	//划分字符串，使其的每一个子串都是回文
	//子集类命题，可以模仿subSet和Combination的递归写法
	class ParadromeSubstr {
	public:
		vector<vector<string>> paradromeAllSubstr(const string& s) {
			vector<vector<string>> vret;
			if (s.empty()) return vret;
			vector<string> cur;
			dfsParadrome(s,0, cur, vret);
			return vret;
		}
	private:
		void dfsParadrome(const string s,int start, vector<string>& cur, vector<vector<string>>& vret) {
			if(start==s.size()){
				vret.push_back(cur);
				return;
			}
			for (int i = start; i < s.size(); ++i) {
				if (isParadrome(s, start, i)) {
					cur.push_back(s.substr(start, i - start + 1));
					dfsParadrome(s, i + 1, cur, vret);
					cur.pop_back();
				}
			}
		}
		bool isParadrome(const string& s, int start, int end) {
			while (start<end&&s[start]==s[end])
			{
				++start;
				--end;
			}
			return start >= end;
		}
	};
	//Unique Paths
	//DP:p[i][j]=p[i-1][j]+p[i][j-1]
	int UniquePathsAmount(int m, int n) {
		vector<vector<int>> vv(m, vector<int>(n, 1));
		for (int i = 1; i < m; ++i)
			for (int j = 1; j < n; ++j)
				vv[i][j] = vv[i - 1][j] + vv[i][j - 1];
		return vv[m-1][n-1];
	}
	//为了节省空间，可以采用逐行刷新
	int uniquePathsAmount(int m, int n) {
		vector<int> f(n, 1);
		for (int i = 1; i < m; ++i)
			for (int j = 1; j < n; ++j)
				f[j] = f[j] + f[j - 1];
		return f[n - 1];
	}
	//Unique Paths II
	//地图中加入了障碍，1表示有障碍，还是求左上点到右下点的路径数量，可以用深搜递归，也可以沿用DP
	int uniquePathsAmountII(const vector<vector<int>>& vv) {
		if (vv.empty() || vv[0].empty() || vv[0][0] == 1) return 0;
		int m = vv.size();
		int n = vv[0].size();
		vector<vector<long long>> dp(m + 1, vector<long long>(n + 1, 0));
		dp[0][1] = 1;
		for(int i=1;i<=m;++i)
			for (int j = 1; j <= n; ++j) {
				if (vv[i - 1][j - 1]) continue;
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		return dp[m][n];
	}
	//N-Queens,Finding All Solutions
	class NQueens {
	public:
		vector<vector<string>> nQueens(int n) {
			vector<vector<string>> vret;
			if (n == 0) return vret;
			vector<int> col(n, -1);
			dfs(0, col, vret);
			return vret;
		}
	private:
		void dfs(int row, vector<int>& col, vector<vector<string>>& vret) {
			int n = col.size();
			if (row == n) {
				vector<string> ss;
				for (int i = 0; i < row; ++i) {
					string s;
					for (int j = 0; j < n; ++j)
						if (j != col[i])
							s.push_back('.');
						else
							s.push_back('Q');
					ss.push_back(s);
				}
				vret.push_back(ss);
				return;
			}
			for (int j = 0; j<n; ++j)
				if (isValid(row, j, col)) {
					col[row] = j;
					dfs(row + 1, col, vret);
				}
		}
		bool isValid(int row, int col, vector<int>& dict) {
			for (int i = 0; i < row; ++i) {
				if (col == dict[i]) return false;
				if (abs(i - row) == abs(dict[i] - col)) return false;
			}
			return true;
		}
	};
	//N-Queens,Finding The Number Of Solutions
	class NQueensAmount {
	public:
		int nQueens(int n) {
			this->amount = 0;
			if (n == 0) return this->amount;
			vector<int> C(n, -1);
			dfs(0, C);
			return this->amount;
		}
	private:
		static int amount;

		void dfs(int row, vector<int>& col) {
			int n = col.size();
			if (row == n) {
				++amount;
				return;
			}
			for (int j = 0; j<n; ++j)
				if (isValid(row, j, col)) {
					col[row] = j;
					dfs(row + 1, col);
				}
		}
		bool isValid(int row, int col, vector<int>& dict) {
			for (int i = 0; i < row; ++i) {
				if (col == dict[i]) return false;
				if (abs(i - row) == abs(dict[i] - col)) return false;
			}
			return true;
		}
	};
	class RestoreIPs {
	public:
		vector<string> restoreIPs(const string& ip) {
			vector<string> vret;
			if (ip.size()<4 || ip.size()>12) return vret;
			vector<string> s;
			dfs(0, ip,s, vret);
			return vret;
		}
	private:
		void dfs(int start, const string& ip, vector<string>&s, vector<string>& vret) {
			if (start == ip.size()) {
				if (s.size() == 4) {
					string ss = "";
					for (auto c : s)
						ss += c;
					ss.pop_back();//弹出末尾的'.'
					vret.push_back(ss);
				}
				return;
			}
			//巧妙的两次剪枝
			if (ip.size() - start > (4 - s.size()) * 3) return;
			if (ip.size() - start < 4 - s.size()) return;

			for (int i = start; i < start+3; ++i) {
				if (isValid(ip.substr(start, i - start + 1))) {
					s.push_back(ip.substr(start, i - start + 1) + '.');
					dfs(i + 1, ip, s, vret);
					s.pop_back();
				}
			}
		}
		bool isValid(string& s) {
			if (s[0] == '0' && s.size()>1) return false;
			int num = 0;
			for (auto c : s)
				num = 10 * num + c - '0';
			if (num > 255)
				return false;
			return true;
		}
	};
	//允许使用重复元素
	//In 2,3,5 target=7 
	//Out [2,2,3],[2,5]
	class CombinationSum {
	public:
		vector<vector<int>> combinationSum(vector<int>& v,int target) {
			vector<vector<int>> vret;
			if (v.empty() || target == 0) return vret;
			sort(v.begin(), v.end());
			vector<int> cur;
			dfs(v, 0, cur, vret,target);
			return vret;
		}
	private:
		void dfs(const vector<int>& v, int start, vector<int>& cur, vector<vector<int>>& vret,int gap) {
			if (gap == 0) {
				vret.push_back(cur);
				return;
			}
			//...
			for (int i = start; i < v.size(); ++i) {
				if (gap < v[i]) return;
				cur.push_back(v[i]);
				dfs(v, i, cur, vret, gap - v[i]);
				cur.pop_back();
			}
		}
	};
	//不允许元素重复使用
	//For example, In:1,2,7,6,1, target=8
	//Out:[1, 7],[2, 6],[1, 1, 6]
	//和上面类似，只要稍微改动，
	//首先在递归的for循环里加上if (i > start && num[i] == num[i - 1]) continue; 这样可以防止res中出现重复项，
	//然后就在递归调用combinationSum2DFS里面的参数换成i+1，这样就不会重复使用数组中的数字了
	class CombinationSumII {
	public:
		vector<vector<int>> combinationSum(vector<int>& v, int target) {
			vector<vector<int>> vret;
			if (v.empty() || target == 0) return vret;
			sort(v.begin(), v.end());
			vector<int> cur;
			dfs(v, 0, cur, vret, target);
			return vret;
		}
	private:
		void dfs(const vector<int>& v, int start, vector<int>& cur, vector<vector<int>>& vret, int gap) {
			if (gap == 0) {
				vret.push_back(cur);
				return;
			}
			//...
			for (int i = start; i < v.size(); ++i) {
				if (gap < v[i]) return;
				if (i > start&&v[i] == v[i - 1]) continue;
				cur.push_back(v[i]);
				dfs(v, i+1, cur, vret, gap - v[i]);
				cur.pop_back();
			}
		}
	};
	class GenerateParenthesis {
	public:
		vector<string> generateParenthesis(int n) {
			vector<string> res;
			generateParenthesisDFS(n, n, "", res);
			return res;
		}
		void generateParenthesisDFS(int left, int right, string out, vector<string> &res) {

			//cout << left << "\t" << right << endl;
			if (left > right) return;
			if (left == 0 && right == 0) res.push_back(out);
			else {
				if (left > 0) generateParenthesisDFS(left - 1, right, out + '(', res);
				if (right > 0) generateParenthesisDFS(left, right - 1, out + ')', res);
			}
		}
	};
	//Sudoku
	//空着的点用'.'表示
	//用一维数组来dfs
	class Sudoku {
	public:
		void sudoku(vector<vector<char>>& vv) {
			vector<char> v;
			for (auto s : vv)
				for (auto c : s)
					v.push_back(c);
			dfs(v,vv, 0);
		}
	private:
		void dfs(vector<char>& v,vector<vector<char>>& vv, int start) {
			for (int i = start; i < v.size(); ++i) {
				if (v[i] != '.') continue;
				for (char c = '1'; c <= '9'; ++c) {
					if (isValid(vv,i, c)) {
						v[i] = c;
						vv[start / 9][start % 9 - 1] = c;
						dfs(v, vv, i + 1);
					}
					v[i] = '.';
					vv[start / 9][start % 9 - 1] = '.';
				}
			}
		}
		bool isValid(vector<vector<char>>& vv,const int pos, const char c) {
			int x = pos / 9;
			int y = pos % 9 - 1;
			for (int j = 0; j < 9; ++j)
				if (vv[x][j] == c) return false;
			for (int i = 0; i < 9; ++i)
				if (vv[i][y] == c) return false;
			for (int i = 3 * x / 3; i < 3 + 3 * x / 3; ++i)
				for (int j = 3 * y / 3; 3 + 3 * y / 3; ++j)
					if (vv[i][j] = c) return false;
			return true;
		}
	};
	//直接dfs
	class SudokuII {
	public:
		bool sudoku(vector<vector<char>>& vv) {
			for (int i = 0; i < 9; ++i)
				for (int j = 0; j < 9; ++j)
					if (vv[i][j] == '.') {
						for (char c = '1'; c <= '9'; ++c) {
							vv[i][j] = c;
							if (isValid(vv, i, j, c) && sudoku(vv))
								return true;
							vv[i][j] = '.';
						}
						return false;
					}
			return true;
		}
	private:
		bool isValid(vector<vector<char>>& vv, int x,int y, const char c) {
			for (int j = 0; j < 9; ++j)
				if (vv[x][j] == c) return false;
			for (int i = 0; i < 9; ++i)
				if (vv[i][y] == c) return false;
			for (int i = 3 * x / 3; i < 3 + 3 * x / 3; ++i)
				for (int j = 3 * y / 3; 3 + 3 * y / 3; ++j)
					if (vv[i][j] = c) return false;
			return true;
		}
	};
}
namespace IterativeDeepening {
	//迭代加深搜索，是以近似广搜的方式进行深搜的方法，当命题适合于广搜求解却又无法承受广搜的空间消耗而时间供给充足时，用迭代加深搜索。
	//层层加深搜索，dfs()中当搜索深度达到给定的maxDepth阈值或找到了解空间时便停止搜索。
	//伪代码类似于如下
	/*for (int i = 0; i < finalDepth; ++i){//除了用finalDepth控制搜索终极深度，还可以设置定时器来控制搜索深度，比如1秒之内，搜索继续加深
		dfs(, 0, i, vret);
		if(timeOut())//检查是否超时
		    break;
		}
	void dfs(,int start , int maxDepth, vector<int>& cur,vector<vector<int>>& vret) {
		if (start > maxDepth)
			return;
		if (找到了解空间) {
			vret.push_back(解空间);
			return;
		}
		for (int i = start; i < 某个值; ++i) {
			cur.push_back(某个节点);
			dfs(, i + 1, maxDepth, cur, vret);
			cur.pop_back();
		}
	}*/
}
namespace ReImplementFunction {
	//Re-Implement pow(x,n)
	//二分法
	class Power {
	public:
		double pow(int x, int n) {
			if (n < 0)
				return 1.0 / power(x, -n);
			else
				return power(x, n);
		}
	private:
		double power(int x, int n) {
			if (n == 0) return 1;
			int v = power(x, n / 2);
			if (n % 2)
				return v*v*x; //包括n=1也融入到这个逻辑中，if (n == 1) return x;
			else
				return v*v;
		}
	};
	//Re-Implement sqrt(x)
	//二分法
	int sqrt(int x) {
		if (x <= 1) return x;
		int left = 0,right = x;
		while (left<right)
		{
			int mid = left + (right - left) / 2;
			if (x / mid > mid)
				left = mid;
			else if (x / mid < mid)
				right = mid;
			else
				return mid;
		}
		return right - 1;
	}
}
namespace GreedyAlgorithm {
	//贪心算法是以局部最优来推导全局最优的方法，但是推导的结果不一定是最优解，可能是近似最优解，绝对最优解的情况很少，例如区间调度安排。
	//集合覆盖问题、旅行商问题，都属于NP完全问题，他们在数学领域上并没有快速得到最优解的方案，贪婪算法是最适合处理这类问题，以得到近似最优解
	//Jump Game
	//从最左边往右跳，在i能跳的距离是a[i]，判断能不能跳到最右边
	//[2,3,1,1,4]->true;[3,2,1,0,4]->false;
	bool jumpGame(const vector<int>& v) {
		if (v.empty()) return false;
		int reach = 0;
		for (int i = 0; i < v.size(); ++i) {
			if (reach < i || reach >= v.size()-1) break;
			reach = max(reach, i + v[i]);
		}
		return reach >= v.size()-1;
	}
	//Jump Game II
	int jumpGameII(const vector<int>& v) {
		int step = 0, pre = 0, cur = 0;
		int i = 0;
		while (cur < v.size()) {
			pre = cur;
			while (i <= pre) {
				cur = max(cur, i + v[i]);
				++i;
			}
			if (cur < i)
				break;
			++step;
		}
		return step;
	}
	//Best Time To Buy And Sell Stock，只能买卖一次
	//时间复杂度O(N)
	//找到最低的点和最高的点，并保证最低点在最高点的前面
	int BuySellStock(const vector<int>& v) {
		//如果是个递减序列，返回-1
		int buy = INT_MAX;
		int profit = 0;
		for (int i = 0; i < v.size(); ++i) {
			buy = min(buy, v[i]);
			profit = max(profit, v[i] - buy);
		}
		return profit;
	}
	//Best Time To Buy And Sell Stock，不限买卖交易次数
	//贪心算法，当明天的价比今天高时，今天买明天卖
	//时间复杂度O(N)
	int BuySellStockII(const vector<int>& v) {
		int profit = 0;
		for (int i = 0; i < (int)v.size() - 1; ++i)//因为是无符号数做减法，先得进行转有符号，用防御式编程先检查容器是否为空也行
			if (v[i] < v[i + 1])
				profit += v[i + 1] - v[i];
		return profit;
	}
	//LongestSubstring WithOut Repeation
	//时间复杂度O(N),空间复杂度O(N)
	int LongestSubstringNoRepeation(const string& s) {
		int left = -1, gap = 0;
		unordered_map<int, int> map;
		for (int i = 0; i < s.size(); ++i) {
			if (map.count(s[i]) && map[s[i]] > left) 
				left = map[s[i]];
			map[s[i]] = i;
			gap = max(gap, i - left);
		}
		return gap;
	}
	//Contain Most Water
	//时间复杂度O(N)，空间复杂度O(1)
	int containMostWater(const vector<int>& v) {
		if (v.empty()) return -1;
		int amount = 0;
		int start = 0, end = v.size() - 1;
		while (start<end)
		{
			amount = max(amount, min(v[start], v[end])*(end - start+1));
			if (v[start] < v[end])
				++start;
			else
				--end;
		}
		return amount;
	}
	//区间重叠问题，包括最多不重叠区间数、最大
	//最多不重叠区间数
	//Greedy
	class Time {
	public:
		int start;
		int end;
		Time(int s, int e) :start(s), end(e) {}
	};
	bool cmp(const Time& t1, const Time& t2) {
		return t1.end < t2.end;
	}
	int countNoOverlap(const vector<int>& vstart, const vector<int>& vend) {
		if (vstart.size() != vend.size() || vstart.empty()) return -1;
		vector<Time> vtime;
		for (int i = 0; i < vstart.size(); ++i)
			vtime.push_back(Time(vstart[i], vend[i]));
		sort(vtime.begin(), vtime.end(),cmp);
		int amount = 0, prevEnd = 0;
		for (int i = 0; i < vtime.size(); ++i) {
			if (prevEnd < vtime[i].start) {
				amount++;
				prevEnd = vtime[i].end;
			}
		}
		return amount;
	}
	//最大不重叠区间和
	//DP
	int noOverlap(const vector<Time>& vtime, int pos, const Time& t) {
		for (int i = pos; i >=0; --i)
			if (vtime[i].end < t.start)
				return i;
		return -1;
	}
	int longestNoOverlap(const vector<int>& vstart, const vector<int>& vend) {
		if (vstart.size() != vend.size() || vstart.empty()) return -1;
		vector<Time> vtime;
		for (int i = 0; i < vstart.size(); ++i)
			vtime.push_back(Time(vstart[i], vend[i]));
		sort(vtime.begin(), vtime.end(), cmp);
		//DP
		int* dp = new int[vtime.size()];
		dp[0] = vtime[0].end - vtime[0].start;
		int longest = dp[0];
		for (int i = 1; i < vtime.size(); ++i) {
			if (int pos = noOverlap(vtime, i, vtime[i]) >= 0)
				longest = dp[pos] + vtime[i].end - vtime[i].start;
			else
				longest = vtime[i].end - vtime[i].start;
			dp[i] = max(dp[i - 1], longest);
		}
		delete[]dp;
		return dp[vtime.size()-1];
	}
	//带权重区间调度，在每个区间上绑定一个权重，求加权之后的区间长度最大值
	//For example,某酒店采用竞标式入住，每一个竞标是一个三元组（开始，入住时间，每天费用）。现在有N个竞标，选择使酒店效益最大的竞标
	//DP,在求最长不重叠的区间时乘以权重
	typedef struct Node {
		int start;
		int end;
		int value;
		Node(int s, int e, int v) :start(s), end(e), value(v) {}
		long long getValue() {
			return value*(end - start);
		}
	} node;
	bool cmpNd(const Node& n1, const Node& n2) {
		return n1.end < n2.end;
	}
	int findNoOverlap(const vector<Node>& wTime, int pos, const Node& node) {
		for (int i = pos; i >= 0; --i)
			if (wTime[i].end < node.start)
				return i;
		return -1;
	}
	int bestValueNoOverlap(const vector<int>& vstart, const vector<int>& vend, const vector<int>& weight) {
		//防御式编程，检查入参的合法性
		//...
		vector<node> wTime;
		for (int i = 0; i < vstart.size(); ++i)
			wTime.push_back(Node(vstart[i], vend[i], weight[i]));
		sort(wTime.begin(), wTime.end(), cmpNd);
		//DP
		long long* dp = new long long[vstart.size()];
		dp[0] = wTime[0].getValue();
		long long best = 0;
		for (int i = 1; i < wTime.size(); ++i) {
			int pos = findNoOverlap(wTime, i, wTime[i]);
			if (pos >= 0)
				best = dp[pos] + wTime[i].getValue();
			else
				best = wTime[i].getValue();
			dp[i] = max(best, dp[i - 1]);
		}
		delete[]dp;
		return dp[wTime.size() - 1];
	}
	//最少区间覆盖
	//用最少的区间覆盖某一给定的区间
	//Greedy
	bool cmp2(const Time& t1, const Time& t2) {
		return t1.start < t2.start;
	}
	int leastRangeToOverlap(vector<Time>& vtime, const int left, const int right,vector<Time>& range) {
		if (vtime.empty() || left >= right) return -1;
		sort(vtime.begin(), vtime.end(), cmp2);
		int n = vtime.size();
		int i = 0;
		int tleft = left;
		int max_right = 0;
		int pos;
		int amount = 0;
		while (i<n)
		{
			while (vtime[i].start <= tleft) {
				if (vtime[i].end > max_right) {
					max_right = vtime[i].end;
					pos = i;
				}
				++i;
			}
			if (max_right > tleft) {
				range.push_back(vtime[pos]);
				amount++;
				tleft = max_right;
				if (tleft >= right)
					return amount;
				++i;
			}
			else
				return -1;
		}
		return amount;
	}
	//最大区间重叠
	//先start排序，从左往右扫描，从相邻的区间找出最大的重叠部分

	//会议安排，着色问题
	//相当于求最大的重叠区间数
	typedef struct  Meet {
		int num;
		int time;//开始或结束时间
		int sore;//开始为1,结束为0
		int room = -1;
		Meet(int n, int t, int f) :num(n), time(t), sore(f) {}
	};
	bool cmp3(const Meet& m1, const Meet& m2) {
		if (m1.time == m2.time) return m1.sore < m2.sore;
		return m1.time < m2.time;
	}
	class MeetSchedule {
	public:
		int meetingRoomSchedule(const vector<int>& vstart, const vector<int>& vend, vector<vector<int>>& vret) {
			//防御式编程
			vector<Meet> vtime;
			for (int i = 0; i < vstart.size(); ++i) {
				vtime.push_back(Meet(i, vstart[i], 1));
				vtime.push_back(Meet(i, vend[i], 0));
			}
			sort(vtime.begin(), vtime.end(), cmp3);
			stack<int> stkroom; //准备教室
			for (int i = vstart.size(); i >= 1; --i)
				stkroom.push(i);
			int maxNum = 0;
			for (int i = 0; i < vtime.size() - 1; ++i) {
				if (vtime[i].sore) {
					maxNum = max(maxNum, stkroom.top());
					vtime[i].room = stkroom.top();
					stkroom.pop();
					setEndRoom(vtime, i, vtime[i]);
					vret.push_back(vector<int>{vtime[i].num, vtime[i].room});
				}
				else
					stkroom.push(vtime[i].room);
			}
			return maxNum;
		}
	private:
		void setEndRoom(vector<Meet>& vtime, int pos, Meet& node) {
			for (int i = pos + 1; i < vtime.size(); ++i)
				if (vtime[i].num == node.num)
					vtime[i].room = node.room;
		}
	};
}
namespace DynamicPlanning
{
	//换钱问题
	//总共有多少种兑换方案
	//Solution 1:深搜，和CombinationSum I 一样，是不需要保存深搜结果的，但是像这样指数级的搜索会超时
	//Solution 2:DP
	int changeMoney(vector<int>& money, int target) {
		if (money.empty() || target <= 0) return -1;
		sort(money.begin(), money.end());
		int n = money.size();
		//DP
		vector<vector<int>> dp(n + 1, vector<int>(target + 1, 0));
		for (int i = 0; i <= n; ++i)
			dp[i][0] = 1;
		for (int i = 1; i <= n; ++i)
			for (int j = 1; j <= target; ++j)
				for (int k = 0; k <= j / money[i-1]; ++k)
					dp[i][j] += dp[i - 1][j - k*money[i-1]];
		return dp[n][target];
	}
	//至少能兑换到几张纸币/枚硬币
	//贪心
	int changeLeastMoney(vector<int>& money, int target, vector<int>& vret) {
		if (money.empty() || target <= 0) return -1;
		sort(money.begin(), money.end(), greater<int>());
		int n = money.size();
		int sum = 0;
		while (target) {
			for (int i = 0; i<n; ++i)
				if (int num = target / money[i]) {
					for (int j = 0; j < num; ++j)
						vret.push_back(money[i]);
					target = target - num*money[i];
					sum += num;
				}
		}
		return sum;
	}
	//兑换固定数量的硬币的方案
	//深搜+剪枝
	void dfs(vector<int>& money, int start, vector<int>& cur, vector<vector<int>>& vret, int gap, int amount) {
		if (cur.size() == amount&&gap == 0) {
			vret.push_back(cur);
			return;
		}
		if (cur.size() > amount)//剪枝
			return;
		for (int i = start; i < money.size(); ++i) {
			if (money[i] > gap) return;
			cur.push_back(money[i]);
			dfs(money, i, cur, vret, gap - money[i], amount);
			cur.pop_back();
		}
	}
	int getSpecificAmount(vector<int>& money, int target, vector<vector<int>>& vret, int amount) {//将target换零，换成amount枚硬币
		 //防御式编程，检查入参
		//...
		sort(money.begin(), money.end());
		vector<int> cur;
		dfs(money, 0, cur, vret, target, amount);
		if (vret.empty())
			return -1;
		return vret.size();
	}
	//复制粘贴问题
	//开始就一个'A',提供2种操作，一是Copy All,将文本框内所有A复制到粘贴板，而是Paste将粘贴板内的A追加到文本框
	//求要使得到n个'A'至少需要进行几次操作
	int KeysKeyBoard(int n,vector<string>& vret) {
		int d = 2;
		vector<int> cur;
		while (n>1)
		{
			while(n%d == 0) {
				cur.push_back(d);
				n /= d;
			}
			++d;
		}
		for (auto i = cur.rbegin(); i != cur.rend(); ++i) {
			vret.push_back("CopyAll");
			for(int j=0;j<*i-1;++j)
				vret.push_back("Paste");
		}
		return accumulate(cur.begin(), cur.end(),0);
	}
	//提供4种操作：按一个A，全选，复制，粘贴
	//可以进行N次操作，求最多可以获得多少个A
	//DP
	int KeysKeyBoardII(int n) {
		//DP
		vector<int> dp(n+1,0);
		for (int i = 1; i <= n; ++i) {
			dp[i] = i;
			for (int j = 1; j <= i - 2; ++j)
				dp[i] = max(dp[i], dp[j] * (i - j - 1));
		}
		return dp[n];
	}
	//Longest Common Subsequence
	//DP Donald.E Knuth
	int LCS(const string& s, const string& t,string& sret) {
		if (s.empty() || t.empty()) return -1;
		int m = s.size();
		int n = t.size();
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		vector<vector<string>> map(m + 1, vector<string>(n + 1, 0));
		for(int i=1;i<=m;++i)
			for (int j = 1; j <= n; ++j) {
				if (s[i - 1] == t[j - 1]) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
					map[i][j] = "\\";
				}
				else if (dp[i-1][j]>=dp[i][j-1])
				{
					dp[i][j] = dp[i - 1][j];
					map[i][j] = "|";

				}
				else {
					dp[i][j] = dp[i][j - 1];
					map[i][j] = "-";
				}
			}
		if (dp[m][n] == 0)
			return 0;
		//回溯出LSC总串
		int i = m, j = n;
		while (i > 0 && j > 0) {
			if (map[i][j] == "\\") {
				sret.push_back(s[i-1]);
				--i;
				--j;
			}
			else if (map[i][j] == "|")
				--i;
			else
				--j;
		}
		reverse(sret.begin(), sret.end());
		return dp[m][n];
	}
	//LIS 最长递增子序列
	//找到该序列可以用LCS的方法，时间复杂度O(N**2)，外加构造一个额外的连续递增字符串
	//若只要求序列长度，则贪心算法,维护一个递增的栈，最后栈的大小就是最长递增序列的长度，时间复杂度O(N*logN)
	int lis(const vector<int>& v) {
		if (v.empty()) return -1;
		vector<int> vstk;
		vstk.push_back(v[0]);
		for (int i = 1; i < v.size(); ++i) {
			if (v[i] > vstk.back())
				vstk.push_back(v[i]);
			else 
				*lower_bound(vstk.begin(), vstk.end(), v[i])=v[i];
		}
		return vstk.size();
	}
	//当容器排过序，lower_bound(x.begin(),x.end(),target)返回第一个大于等于target的位置，upper_bound是返回第一个大于target的位置；
	//lower_bound和prev一起使用可以找第一个小于的位置，prev和upper_bound可以找第一个小于等于的位置
	
	//最大连续子序列和
	//时间复杂度O(N)，空间复杂度O(1)
	int MaxSumOfSubsequence(const vector<int>& v) {
		if (v.empty()) return -1;
		int maxSum = INT_MIN;
		int curmax = 0;
		for (int i = 0; i < v.size(); ++i) {
			curmax = max(curmax + v[i], v[i]);
			//当curmax>0时，curmax=curmax+v[i];
			//当curmax<0时，curmax=v[i];
			maxSum = max(maxSum, curmax);
		}
		return maxSum;
	}
	//最大连续子序列积
	//考虑负负得正
	//时间复杂度O(N)，空间复杂度O(1)
	int MaxMutilOfSubsequence(const vector<size_t>& v) {
		size_t maxMultil = v[0];
		size_t minCur = v[0];
		size_t maxCur = v[0];
		for (int i = 1; i < v.size(); ++i) {
			size_t end1 = maxCur*v[i];
			size_t end2 = minCur*v[i];
			maxCur = max(max(end1,end2),v[i]);
			minCur = min(min(end1,end2),v[i]);
			maxMultil = max(maxMultil, maxCur);
		}
		return maxMultil;
	}
	//Matrix Chain Multiply
	//求最小的计算代价，比如对于维数分别为10*100、100*5、5*50的矩阵A、B、C，(A*B)*C的代价是7500次计算，优于A*(B*C)的代价75000次计算。
	int MaxtrixChainMultiply(const vector<pair<int, int>>& vp) {//pair<int,int>存储了矩阵的行列
		if (vp.size() < 2) return 0;
		int n = vp.size();
		//预处理，将行列提取到一维数组
		vector<int> v(n + 1, 0);
		for (int i = 0; i < n;++i) {
			v[i] = vp[i].first;
			v[i+1] = vp[i].second;
		}
		//DP
		vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
		for(int len=2;len<=n;++len)
			for (int i = 1; i <= n - len + 1; ++i) {
				dp[i][i] = 0;
				int j = i + len - 1;
				dp[i][j] = INT_MAX;
				for (int k = i; k < j; ++k)
					dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + v[i - 1] * v[k] * v[j]);
			}
		return dp[1][n];
	}
	//Advanced,找到具体的划分方法
	class MatrixChainMultiply {
	public:
		int MaxtrixChainMultiplyII(const vector<pair<int, int>>& vp) {//pair<int,int>存储了矩阵的行列
			if (vp.size() < 2) return 0;
			int n = vp.size();
			//预处理，将行列提取到一维数组
			vector<int> v(n + 1, 0);
			for (int i = 0; i < n; ++i) {
				v[i] = vp[i].first;
				v[i + 1] = vp[i].second;
			}
			//DP
			vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
			vector<vector<int>> path(n + 1, vector<int>(n + 1, 0));//记录划分点
			for (int len = 2; len <= n; ++len)
				for (int i = 1; i <= n - len + 1; ++i) {
					dp[i][i] = 0;
					int j = i + len - 1;
					dp[i][j] = INT_MAX;
					for (int k = i; k < j; ++k) {
						int cur = dp[i][k] + dp[k + 1][j] + v[i - 1] * v[k] * v[j];
						if (cur < dp[i][j]) {
							dp[i][j] = cur;
							path[i][j] = k;
						}
					}
				}
			printPath(path, 1, n);
			return dp[1][n];
		}
	private:
		void printPath(vector<vector<int>>& path, int i, int j) {
			if (i == j)
				cout << "M" << i;
			else {
				cout << "(";
				printPath(path, i, path[i][j]);
				printPath(path,path[i][j]+1,j);
				cout << ")";
			}
		}
	};
	//背包问题,01,部分，完全
	//01
	class Bag01 {
	public:
		int bag01(const int cap, const vector<int>& volume, const vector<int>& value) {
			//预处理
			int n = volume.size();
			vector<int> vol(volume);
			vector<int> val(value);
			vol.insert(vol.begin(), 0);
			val.insert(val.begin(), 0);
			//DP
			vector<vector<int>> dp(n + 1, vector<int>(cap + 1, 0));
			for (int i = 1; i <= n; ++i)
				for (int j = 1; j <= cap; ++j)
					if (vol[i] > j)
						dp[i][j] = dp[i - 1][j];
					else
						dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - vol[i]] + val[i]);
			//打印所选的物品
			print(dp, vol, n, cap);
			return dp[n][cap];
		}
	private:
		void print(const vector<vector<int>>& dp, const vector<int>& vol, int i, int j) {//完全根据求解dp[i][j]来回溯
			if (i <= 0)
				return;
			if (dp[i][j] == dp[i - 1][j])
				print(dp, vol, i - 1, j);
			else {
				print(dp, vol, i - 1, j - vol[i]);
				cout << i << "\t";
			}
		}
	};
	//partion Bag
	int partionBag(const int cap, const vector<int>& amount, const vector<int>& volume, const vector<int>& value) {
		//防御式编程，检查入参合法性
		//...
		//预处理
		int n = amount.size();
		vector<int> amt(amount);
		vector<int> vol(volume);
		vector<int> val(value);
		amt.insert(amt.begin(), 0);
		vol.insert(vol.begin(), 0);
		val.insert(val.begin(), 0);
		//DP
		vector<vector<int>> dp(n + 1, vector<int>(cap + 1, 0));
		for (int i = 1; i <= n; ++i)
			for (int j = 1; j <= cap; ++j)
				if (vol[i] > j)
					dp[i][j] = dp[i - 1][j];
				else {
					dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - vol[i]]+val[i]);
					for (int k = 1; k <= amt[i] && k*vol[i] <= j; ++k)
						dp[i][j] = max(dp[i][j], dp[i - 1][j - k*vol[i]] + k*val[i]);
				}
		return dp[n][cap];
	}
	//完全
	int entireBag(const int cap, const vector<int>& volume, const vector<int>& value) {
		//防御式编程，检查入参合法性
		//...
		//预处理
		int n = volume.size();
		vector<int> vol(volume);
		vector<int> val(value);
		vol.insert(vol.begin(), 0);
		val.insert(val.begin(), 0);
		//DP
		vector<vector<int>> dp(n + 1, vector<int>(cap + 1, 0));
		for (int i = 1; i <= n; ++i)
			for (int j = 1; j <= cap; ++j)
				if (vol[i] > j)
					dp[i][j] = dp[i - 1][j];
				else {
					dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - vol[i]] + val[i]);
					for (int k = 1; k*vol[i] <= j; ++k)
						dp[i][j] = max(dp[i][j], dp[i - 1][j - k*vol[i]] + k*val[i]);
				}
		return dp[n][cap];
	}
	//瓷砖覆盖（状态压缩DP）
	//难 https://www.cnblogs.com/wuyuegb2312/p/3281264.html
	//状态压缩+深搜+DP
	class CoverTileSolution {
	public:
		int coverTile(int m, int n) {
			if (n > m)//m行n列,列取小
				swap(m, n);
			vector<vector<int>> dp(m, vector<int>(1 << n, 0));
			dfs(0, 0, n, dp, 0, 1);
			for (int i = 1; i<m; ++i)
				for (int j = 0; j<(1 << n); ++j)
					if (dp[i - 1][j])
						dfs(i, 0, n, dp, (~j)&((1 << n) - 1), dp[i - 1][j]);
			return dp[m - 1][(1 << n) - 1];
		}
	private:
		void dfs(int row, int col, int n, vector<vector<int>>& dp, int status, int preRowStatus) {
			if (col == n) {
				dp[row][status] += preRowStatus;
				return;
			}
			dfs(row, col + 1, n, dp, status, preRowStatus);//不放
			if (col > n - 2 || (1 << col)&status || (1 << (col + 1))&status) return;//剪枝
			dfs(row, col + 2, n, dp, status | (1 << (col + 1)) | (1 << col), preRowStatus);
		}
	};
	//线性分割 Liner partision)
	//将一组数以相对顺序不变的方式分成K份，使K份数据的总量均匀，和尽量接近，(即最大的那一份尽量小)
	//Solution I:深搜+剪枝,保留最均匀的那一份
	//Solution II:DP

	//3次捡苹果，可以转化为3次1回的捡苹果
	//欧几里得旅行商问题可以求到最短闭合路程，需要多项式时间复杂度，可以转化为双调欧几里得旅行商问题，可以在O(N**2)时间复杂度内DP求解.
	//剪绳子,剪成k段,(k>=1)，返回最大乘积
	//Solution I: DP:dp[i]=dp[j]*dp[i-j](1<=j<=i/2)
	int cutRope(const int length) {
		if (length == 0) return -1;
		//DP
		vector<int> dp(length + 1, 0);
		dp[1] = 1;
		dp[2] = 2;
		dp[3] = 3;
		for (int i = 4; i <= length; ++i)
			for (int j = 1; j <= i / 2; ++j)
				dp[i] = max(dp[i], dp[j] * dp[i - j]);
		return dp[length];
	}
	//Solution II:Greedy Algorithm
	//一个数只有当拆到3的个数达到最大时它的拆乘才会最大，但如果结尾剩下1的话要少拆1个3从而拼成1个4
	int cutRopeII(const int length) {
		if (length == 0) return -1;
		if (length < 4) return length;
		if (length % 3 == 1)
			return pow(3, length / 3 - 1) * 4;
		else
			return pow(3, length / 3)*(length % 3 == 0 ? 1 : 2);
	}
	//paint fence
	//n个篱笆，k种颜色，不出现连续三个篱笆涂相同的颜色
	int painFence(const int n, const int k) {
		if (n == 0 || k == 0) return 0;
		//DP
		vector<int> dp(n + 1, 0);
		dp[1] = k;
		dp[2] = k*k;
		for (int i = 3; i <= n; ++i)
			dp[i] = dp[i - 2] * (k - 1) + dp[i - 1] * (k - 1);
		return dp[n];
	}
	//Triangle
	int TriangleMinPath(vector<vector<int>>& vv) {
		if (vv.empty() || vv[0].empty()) return INT_MAX;
		for(int i=0;i<vv.size();++i)
			for (int j = 0; j < vv[i].size(); ++j)
				if (j == 0)
					vv[i][j] = vv[i - 1][j];
				else if (j = vv[i].size() - 1)
					vv[i][j] = vv[i - 1][j-1];
				else
					vv[i][j] += min(vv[i - 1][j - 1], vv[i - 1][j]);
		return *min_element(vv[vv.size() - 1].begin(), vv[vv.size() - 1].end());
	}
	//Palindrome Substring II
	class PalindromeSubstring {
	public:
		int palindromeSubstring(const string& s) {
			if (s.empty()) return -1;
			//DP
			int n = s.size();
			vector<int> dp(n, 0);
			dp[0] = 1;
			for (int i = 1; i < n; ++i) {
				dp[i] = i;
				for(int j=0;j<i;++j)
					if (isPalindrome(s, j + 1, i))
						dp[i] = min(dp[i], dp[j] + 1);
			}
			return dp[n - 1];
		}
	private:
		bool isPalindrome(const string& s, int left, int right) {
			while (s[left] == s[right]) {
					++left; --right;
			}
			return left >= right;
		}
	};
	//Interleaving String
	//判断s3是不是由s2和s1镶嵌组合而成,且s3.size()==s2.size()+s1.size()s2,
	//DP
	bool interleavingString(const string& s1, const string& s2, const string& s3) {
		if (s3.size() != s2.size() + s1.size()) return false;
		if (!s1.size() && !s2.size() && !s3.size()) return true;
		//DP
		int m = s1.size(), n = s2.size();
		vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
		dp[0][0] = true;
		for (int i = 1; i <= m; ++i)
			dp[i][0] = dp[i-1][0]&&s1[i-1] == s3[i-1];
		for (int j = 1; j <= n; ++j)
			dp[0][j] = dp[0][j-1] && s2[j-1] == s3[j-1];
		for (int i = 1; i <= m; ++i)
			for (int j = 1; j <= n; ++j)
				dp[i][j] = (dp[i - 1][j] && s1[i - 1] == s3[i + j - 1]) || (dp[i][j - 1] && s2[j - 1] == s3[i + j - 1]);
		return dp[m][m];
	}
	//Edit Distance
	//DP
	int editDistance(const string& s1, const string& s2) {
		int m = s1.size(), n = s2.size();
		//DP
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		for (int i = 0; i <= m; ++i)
			dp[i][0] = i;
		for (int j = 0; j <= n; ++j)
			dp[0][j] = j;
		for (int i = 1; i <= m; ++i)
			for (int j = 1; j <= n; ++j)
				if (s1[i - 1] == s2[j - 1])
					dp[i][j] = dp[i - 1][j - 1];
				else
					dp[i][j] = min(dp[i - 1][j - 1], min(dp[i][j - 1], dp[i - 1][j]))+1;
		return dp[m][n];
	}
	//Decode Ways
	//'A' -> 1
	//'B' -> 2
	//	...
	//	'Z' -> 26
	//Given encoded message "12" , it could be decoded as "AB" (1 2) or "L" (12).The number of ways decoding "12" is 2
	//Solution I: 深搜,还可以保存所有可能结果
	//Solution II:类似于斐波拉契楼梯,DP求解
	int decodeWays(const string& s) {
		if (s.empty()) return -1;//入参检查中，可以不用检查首字符是否为‘0’，因为两种特殊串如"012356"和"123560"最后Ways都会判0
		//DP
		vector<int> dp(s.size()+1, 1);
		for (int i = 1; i <= s.size(); ++i) {
			dp[i] = s[i - 1] == '0' ? 0 : dp[i - 1];
			if (i > 1 && (s[i - 2] == '1' || (s[i - 2] == '2'&&s[i - 1] <= '6')))
				dp[i] += dp[i - 2];
		}
		return dp[s.size()];
	}
	//Distinct Subsequence
	//先画出例子的DP表，类似于找规律，推出转移公式
	int distinctSubsequence(const string& s, const string& t) {
		if (s.empty() && !t.empty()) return -1;
		if (t.empty()) return 1;
		//DP
		int m = t.size(), n = s.size();
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		for (int j = 0; j <= n; ++j)
			dp[0][j] = 1;
		for (int i = 1; i <= m; ++i)
			for (int j = 1; j <= n; ++j)
				if (t[i - 1] == s[j - 1])
					dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1];
				else
					dp[i][j] = dp[i][j - 1];
		return dp[m][n];
	}
	//Word BreakI
	//Solution I:深搜
	//Solution II:DP
	bool wordBreak(const string& s, const unordered_set<string>& dict) {
		if (s.empty()) return true;
		//DP
		vector<bool> dp(s.size() + 1, false);
		dp[0] = true;
		for (int i = 0; i <= s.size(); ++i)
			for (int j = 0; j < i; ++j)
				if (dp[j] && dict.count(s.substr(j, i - j))) {
					dp[i] = true;
					break;
				}
		return dp[s.size()];
	}
	//Word Break II
	//返回所有可能的组合
	//Solution:深搜
	class WordBreak {
	public:
		vector<string> wordBreakII(const string& s, const unordered_set<string>& dict) {
			vector<string> vret;
			if (s.empty()) return vret;
			vector<string> cur;
			dfs(s, 0, dict, cur,vret);
			return vret;
		}
	private:
		void dfs(const string& s, int start, const unordered_set<string>& dict, vector<string>& cur, vector<string>& vret) {
			if (start == s.size()) {
				string t;
				for (auto c : cur)
					t += (c + " ");
				t.pop_back();
				vret.push_back(t);
				return;
			}
			for (int i = start; i < s.size(); ++i) {
				cur.push_back(s.substr(start, i - start + 1));
				if (dict.count(s.substr(start, i - start + 1)))
					dfs(s, i + 1, dict, cur, vret);
				cur.pop_back();
			}
		}
	};
}
namespace Graphics{}
namespace ClassicalSortions {
	//选择、插入、冒泡、快速、归并、希尔、堆、计数、基数、桶排序 这10种经典的排序算法及其衍生算法，得烂熟于心
	//算法性能评判的准则包括时间复杂度、空间复杂度、常数项、稳定性
	//排序算法分为线性算法和非线性算法，把时间复杂度为O(N)的算法称为线性算法
	//各算法的公认性能参照 https://blog.csdn.net/opooc/article/details/80994353
	//N!>x^n>...>3^n>2^n>N^x>...>N^3>N^2>NlogN>N>logN>1
	/*
	数据是随机整数，时间单位是秒
	数据规模|快速排序 归并排序 希尔排序 堆排序
	1000万 |  0.75  1.22    1.77    3.57
	5000万 |  3.78  6.29    9.48    26.54
	1亿    |  7.65  13.06   18.79   61.31
	*/
	//1.系统的sort 方法，发现传进来的值为数值型，会使用快排，如果发现传的还有比较器，会使用归并排序
	//2.归并和快排哪种更快？
	//	快排比归并排序的常数项要低，所以要快。
	//3.为什么会有归并和快排两种呢？
	//	在比较的时候，使用比较器的时候，要追求一个稳定性，使用 归并排序 可以达稳定性的效果；使用快排不能够实现稳定性的效果。
	//4.面对大规模的时候，当排序量是小于等于60的时候，sort方法 会在内部使用插入排序的方法（不一定是60，是一定的规模）当数据量很低的时候，插入排序的常数项低。
	//5.在c语言中有一版，把归并排序，改成非递归，是基于工程其他考虑。

	class ClassicalSortions {
	public:
		//经典冒泡
		void bubble(int arr[], int len) {
			for (int i = 0; i < len; ++i)
				for (int j = 1; j < len - i; ++j)
					if (arr[j - 1] > arr[j]) exchange(arr, j - 1, j);
		}
		//+外层优化冒泡
		//用一个flag来判断一下，当前数组是否已经有序
		void bubbleI(int arr[], int len) {
			for (int i = 0; i < len; ++i) {
				bool isOrder = true;
				for (int j = 1; j < len - i; ++j)
					if (arr[j - 1] > arr[j]) {
						exchange(arr, j - 1, j);
						isOrder = false;
					}
				if (isOrder) return;
			}
		}
		//+内层优化冒泡
		//用一个pos来记录上一轮冒泡中最后发生交换的位置
		void bubbleII(int arr[], int len) {
			int k = len;
			int lastPos = 0;
			for (int i = 0; i < len; ++i) {
				bool isOrder = true;
				for(int j=1;j<k;++j)
					if (arr[j - 1] > arr[j]) {
						exchange(arr, j - 1, j);
						isOrder = false;
						lastPos = j;
					}
				if (isOrder) return;
				k = lastPos;
			}
		}
		void insert(int arr[],int len) {
			for (int i = 1; i < len; ++i)
				for (int j = i; j >= 0; --j)
					if (arr[j - 1] > arr[j]) exchange(arr, j - 1, j);
		}
		void select(int arr[],int len) {
			for (int i = 0; i < len; ++i) {
				int minIdx = i;
				for (int j = i + 1; j < len; ++j)
					if (arr[minIdx] > arr[j]) minIdx = j;
				exchange(arr, i, minIdx);
			}
		}
		void quick(int arr[], int len) {
			if (len < 2) return;
			quickSort(arr, 0, len - 1);
		}
		void mergeSort(int arr[], int len) {
			if (len < 2) return;
			mergeSort(arr, 0, len - 1);
		}
		//插入排序的改进，第一个突破O(N**2)时间复杂度的排序算法
		void shell(int arr[]);
		//根大堆排序法
		void heap(int arr[], int len) {
			if (len < 2) return;
			for (int i = 0; i < len; ++i)
				buildHeap(arr, i);//建堆
			int tail = len - 1;
			exchange(arr, 0, tail);
			while (tail > 1) {
				keepHeap(arr, 0, tail--);//维护堆
				exchange(arr, 0, tail);
			}
		}
		//计数排序算法，在最大数不大且比较集中时比较有效
		void count(int arr[], int len) {
			if (len < 2) return;
			int maxNum = arr[0];
			for (int i = 1; i < len; ++i)
				if (arr[i] > maxNum) maxNum = arr[i];
			int* t = new int[maxNum + 1]();//假设数均不是负数
			for (int i = 0; i < len; ++i) t[arr[i]]++;
			int j = 0;
			for(int i=0;i<=maxNum;++i)
				while (int count = t[i] > 0) {
					arr[j++] = i;
					t[i]--;
				}
			delete[]t;
		}
		//性能不如桶排序，但却比桶排序更适合海量数据排序，还适合非数值键值排序
		void base(int arr[], int len) {
			int max = arr[0];
			for (int i = 1; i < len; ++i)
				if (arr[i] > max) max = arr[i];
			int bits = 1;
			while (max = max / 10 > 0) ++bits;
			vector<vector<int>> bucket(10, vector<int>());
			for (int exp = 1; exp < pow(10, bits - 1); exp *= 10) {
				for (int j = 0; j < len; ++j)
					bucket[arr[j] / exp % 10].emplace_back(arr[j]);
				int k = 0;
				for (int i = 0; i < 10; ++i)
					for (int j = 0; j < bucket[i].size(); ++j)
						arr[k++] = bucket[i][j];
			}
		}
		//当待排序的一组数均匀独立的分布在一个范围中时用桶排序比较合适
		void barral(int arr[], int len, int size) {
			if (len < 2) return;
			int minV = arr[0], maxV = arr[0];
			for (int i = 1; i < len; ++i) {
				if (arr[i] < minV) minV = arr[i];
				if (arr[i] > maxV) maxV = arr[i];
			}
			if (minV == maxV) return;
			int barAmount = (maxV - minV + 1) / size;
			vector<vector<int>> barral(barAmount + 1, vector<int>());
			for (int i = 0; i < len; ++i) {
				int barId = arr[i] / 3;
				barral[barId].emplace_back(arr[i]);
			}
			for (int i = 0; i < barral.size(); ++i)
				sort(barral[i].begin(), barral[i].end());
			int index = 0;
			for (auto v : barral)
				if (!v.empty())
					for (auto mel : v) arr[index++] = mel;
		}
	private:
		//Genel Functions
		template<typename T> int len(const T& arr) {
			return (int)sizeof(arr) / sizeof(arr[0]);
		}
		template<typename T> void exchange(T arr[], int a, int b) {
			arr[a] = arr[a] ^ arr[b];
			arr[b] = arr[a] ^ arr[b];
			arr[a] = arr[a] ^ arr[b];
		}
		template<typename T> bool isEqual(const T a[], const int lena, const T b[], const int lenb) {
			if (a == nullptr && b == nullptr) return true;
			if (a == nullptr || b == nullptr) return false;
			if (lena != lenb) return false;
			for (int i = 0; i < lena; ++i)
				if (a[i] != b[i]) return false;
			return true;
		}
		template<typename T> T* copyArray(const T arr[], int length) {
			T* ret = new T[length];
			for (int i = 0; i < length; ++i)
				ret[i] = arr[i];
			return ret;
		}
		//quickSort
		void quickSort(int arr[], int left, int right) {
			if (left < right) {
				srand(time(0));
				int random = rand() % (right - left + 1) + left;
				exchange(arr, random, right);
				int pivot = separate(arr, left, right);
				quickSort(arr, left, pivot - 1);
				quickSort(arr, pivot + 1, right);
			}
		}
		int separate(int arr[], int left, int right) {
			int base = arr[right];
			int i = left;
			int j = right;
			while (i < j) {
				while (i < j&&arr[i] < base) ++i;
				arr[j] = arr[i];
				while (i<j&&arr[j]>base) --j;
				arr[i] = arr[j];
			}
			arr[i] = base;
			return i;
		}
		//mergeSort
		void mergeSort(int arr[], int left, int right) {
			if (left < right) {
				int mid = left + (right - left) / 2;
				mergeSort(arr, left, mid - 1);
				mergeSort(arr, mid, right);
				merge(arr, left, mid, right);
			}
		}
		void merge(int arr[], int left, int mid, int right) {
			int* t = new int[right - left + 1];
			int i = left, j = mid;
			int k = 0;
			while (i < mid&&j <= right)
				t[k++] = arr[i] > arr[j] ? arr[i++] : arr[j++];
			while (i < mid) t[k++] = arr[i++];
			while (j <= right) t[k++] = arr[j++];
			for (int i = 0; i < right - left + 1; ++i) arr[left + i] = t[i];

		}
		//往上走建堆
		void buildHeap(int arr[], int index) {
			while (arr[index] > arr[(index - 1) / 2]) {
				exchange(arr, (index - 1) / 2, index);
				index = (index - 1) / 2;
			}
		}
		//往下走维护堆，时间复杂度仅为log(N)
		void keepHeap(int arr[], int index, int size) {
			while (index * 2 + 1 < size) {//因为是完全二叉树结构，只要保证节点的左孩子有效既可，左孩子无效右孩子一定无效
				int leftChild = index * 2 + 1;
				int maxChild = leftChild;
				if (leftChild + 1 < size&& arr[leftChild + 1]>arr[leftChild]) maxChild = leftChild + 1;
				if (arr[index] >= arr[maxChild]) return;
				exchange(arr, index, maxChild);
				index = maxChild;
			}
		}
	};
	//Top K 问题
	//从几百万条数据中找出Top K 条数据
	//O(N*logK)
	class TopK {
	public:
		template<typename T> vector<T> topK(T arr[], int len, int k) {
			//建立k堆，求最小集合则建立最大堆，反之建最小堆
			//这里假设建最小堆
			vector<T> heap;
			heap.reserve(k);
			for (int i = 0; i < k; ++i) heap.emplace_back(arr[i]);
			build_heap(heap, k);
			for (int i = k; i<len; ++i)
				if (arr[i] > heap[0]) {
					heap[0] = arr[i];
					keep_heap(heap, k);
				}
			return heap;
		}
	private:
		template<typename T> void build_heap(vector<T>& arr, int k) {
			for (int i = 0; i < k; ++i) {
				int index = i;
				while (arr[index] < arr[(index - 1) / 2]) {
					swap(arr[index], arr[(index - 1) / 2]);
					index = (index - 1) / 2;
				}
			}
		}
		template<typename T> void keep_heap(vector<T>& arr, int k) {
			int index = 0;
			while (int leftChild = 2 * index + 1 < k) {
				int minChild = leftChild;
				if (leftChild + 1 < k)
					minChild = arr[leftChild + 1] < arr[leftChild] ? leftChild + 1 : leftChild;
				if (arr[index] <= arr[minChild]) return;
				swap(arr[minChild], arr[index]);
				index = minChild;
			}
		}
	};
}
namespace OfferGot {//剑指offer第二版，细节题
	//顺时针打印矩阵
	class PrintMatrixClockwisely {
	public:
		void printMatrixClockwisely(const vector<vector<int>>& vv) {
			if (vv.empty()||vv[0].empty()) return;
			int m = vv.size(), n = vv[0].size();
			int start = 0;
			while (m > 2 * start&&n > 2 * start) {
				printClockwiselyOnce(vv, m, n, start);
				++start;
			}
		}
	private:
		void printClockwiselyOnce(const vector<vector<int>>& vv, int rows, int cols, int start) {
			int endX = rows - 1 - start;
			int endY = cols - 1 - start;
			//从左向右打印
			for (int j = start; j <= endY; ++j)
				cout << vv[start][j] << "\t";
			//从上向下打印
			if (endX > start) {
				for (int i = start + 1; i <= endX; ++i)
					cout << vv[i][endY] << "\t";
			}
			//从右往左打印
			if (endX > start&& start < endY) {
				for (int j = endY - 1; j >= start; --j)
					cout << vv[endX][j] << "\t";
			}
			//从下往上打印
			if (start < endY&&start < endX - 1) {
				for (int i = endX - 1; i > start; --i)
					cout << vv[i][start] << "\t";
			}
		}
	};
	//两个链表的第一个公共节点
	struct ListNode {
		int val;
		ListNode* next = nullptr;
		ListNode(int value):val(value){}
	};
	int getLinkListLength(ListNode* head) {
		int length = 0;
		ListNode* cur = head;
		while (cur) {
			++length;
			cur = cur->next;
		}
		return length;
	}
	ListNode* firstCommonNode(ListNode* head1, ListNode* head2) {
		if (head1 == nullptr || head2 == nullptr) return nullptr;
		int length1 = getLinkListLength(head1);
		int length2 = getLinkListLength(head2);
		ListNode* longCur, *shortCur;
		int lengthDif = 0;
		if (length1 > length2) {
			lengthDif = length1 - length2;
			longCur = head1;
			shortCur = head2;
		}
		else {
			lengthDif = length2 - length1;
			longCur = head2;
			shortCur = head1;
		}
		for (int i = 0; i < lengthDif; ++i)
			longCur = longCur->next;
		while (longCur&&shortCur&&longCur != shortCur) {
			longCur = longCur->next;
			shortCur = shortCur->next;
		}
		return longCur;
	}
	//二叉树和双向链表，将一个二叉排序树转化为双向链表，不用额外空间
	struct TreeNode {
		int value;
		TreeNode* left = nullptr;
		TreeNode* right = nullptr;
		TreeNode(int val):value(val){}
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
	//二叉树的下一个节点
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
	//双队列实现栈
	template<typename T>
	class CQueue {
	public:
		void appendTail(const T& node) {
			stk1.push(node);
		}
		T deleteHead() {
			if (stk2.empty()) 
				while (!stk1.empty()) {
					T top = stk1.top();
					stk2.push(top);
					stk1.pop();
				}
			if (stk2.empty()) throw new exception("queue is empty");
			T ret = stk2.top();
			stk2.pop();
			return ret;
		}
	private:
		stack<T> stk1;
		stack<T> stk2;
	};
	//双栈实现队列
	template<typename T>
	class Qstack {
	public:
		void push(const T& node) {
			if (que1.empty() && que2.empty()) que1.push(node);
			else if (!que1.empty()) que1.push(node);
			else if (!que2.empty()) que2.push(node);
		}
		T pop() {
			if (que1.empty() && que2.empty()) throw new exception("stack is empty");
			if (!que1.empty()) {
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
		queue<int> que1;
		queue<int> que2;
	};
	//打印从1到N位最大的数
	//Solution I:用字符串模拟加一运算
	void printToMaxNums(int n) {
		string s = "0";
		for (;;) {
			int carry = 1;
			for (auto i = s.rbegin(); i != s.rend(); ++i) {
				unsigned int sum = *i - '0' + carry;
				carry = sum / 10;
				*i = sum % 10 + '0';
				if (carry == 0) break;
			}
			if (carry > 0) s.insert(s.begin(), '1');
			if (s.size() > n) break;
			cout << s << endl;
		}
	}

	//Solution II:递归构造全排列
	class PrintToMaxNumsII {
	public:
		void printToMaxNumsII(int n) {
			string s(n, '0');
			permutateRecursively(s, 0, n);
		}
	private:
		void permutateRecursively(string& s, int pos, int length) {
			if (pos == length) {
				printstringNum(s);
				return;
			}
			for (int i = 0; i < 10; ++i) {
				s[pos] = i + '0';
				permutateRecursively(s, pos + 1, length);
			}
		}

		void printstringNum(string& s) {
			auto i = s.begin();
			for (; i != s.end(); ++i)
				if (*i > '0') break;
			if (i == s.end()) return;
			cout << string(i, s.end()) << endl;
		}

	};

	//大整数加减法运算
	//双正、双负、一正一负
	class PlusBigNumbers {
	public:
		string plusTwoBigNum(string s1, string s2) {
			if (s1 == "" || s2 == "") return "empty Integer";
			if (!checkInput(s1) || !checkInput(s2)) return "invalid Integer";
			if (s1[0] == '+'&&s2[0] == '+')
				return plusTwoPlusNum(string(next(s1.begin()), s1.end()), string(next(s2.begin()), s2.end()));
			else if ((s1[0] == '+'&&s2[0] != '-') || (s1[0] != '-'&&s2[0] == '+')) {
				s1 = s1[0] == '+' ? string(next(s1.begin()), s1.end()) : s1;
				s2 = s2[0] == '+' ? string(next(s2.begin()), s2.end()) : s2;
				return plusTwoPlusNum(s1, s2);
			}
			else if (s1[0] == '-'&&s2[0] == '-') {
				string sret = plusTwoPlusNum(string(next(s1.begin()), s1.end()), string(next(s2.begin()), s2.end()));
				sret.insert(sret.begin(), '-');
				return sret;
			}
			else if ((s1[0] >= '0'&&s1[0] <= '9') && (s2[0] >= '0'&&s2[0] <= '9'))
				return plusTwoPlusNum(s1, s2);
			else
				return subTwoNums(s1, s2);
		}
	private:
		bool checkInput(const string& s) {
			if ((s[0]<'0' || s[0]>'9') && (s[0] != '+' && s[0] != '-')) return false;
			if (s == "+" || s == "-") return false;
			auto i = next(s.begin());
			for (; i != s.end(); ++i)
				if (*i<'0' || *i>'9')
					return false;
			return true;
		}
		string plusTwoPlusNum(string& s1, string& s2) {
			string s;
			auto i = s1.rbegin();
			auto j = s2.rbegin();
			int carry = 0;
			while (i != s1.rend() || j != s2.rend()) {
				int a = i == s1.rend() ? 0 : *i - '0';
				int b = j == s2.rend() ? 0 : *j - '0';
				int sum = a + b + carry;
				carry = sum / 10;
				s.insert(s.begin(), sum % 10 + '0');
				i = i == s1.rend() ? i : ++i;
				j = j == s2.rend() ? j : ++j;
			}
			if (carry > 0) s.insert(s.begin(), '1');
			return s;
		}
		string subTwoNums(string& s1, string& s2) {
			bool isPlus = true;
			if (!adjust(s1, s2)) {
				isPlus = false;

				string s = s1;
				s1 = s2;
				s2 = s;
			}
			auto i = s1.rbegin();
			auto j = s2.rbegin();
			int carry = 0;
			int result;
			while (i != s1.rend() || j != s2.rend()) {
				int a = i == s1.rend() ? 0 : *i - '0';
				int b = j == s2.rend() ? 0 : *j - '0';
				if (a < b) {
					result = 10 + a - b - carry;
					carry = 1;
				}
				else {
					result = a - b - carry;
					carry = 0;
				}
				*i = result + '0';
				i = i == s1.rend() ? i : ++i;
				j = j == s2.rend() ? j : ++j;
			}
			if (!isPlus) return '-' + s1;
			return s1;
		}
		bool adjust(string& s1, string& s2) {
			if (s1[0] == '-') {
				string s = s1;
				s1 = s2;
				s2 = s;
			}
			s1 = s1[0] == '+' ? string(next(s1.begin()), s1.end()) : s1;
			s2 = string(next(s2.begin()), s2.end());
			cout << s1 << "\t" << s2 << endl;
			if (s1.size() > s2.size()) return true;
			else if (s1.size() < s2.size()) return false;
			else
				for (int i = 0; i < s1.size(); ++i)
					if (s1[i] > s2[i])
						return true;
			return false;
		}
	};
	//大整数的乘法运算

}