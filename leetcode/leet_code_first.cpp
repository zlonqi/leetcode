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
#include<assert.h>

//��һ��
namespace Array {
	//March_3 �Ƴ��������ظ���Ԫ��
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
	int  removeDuplicates(vector<int>& v) {
		if (v.size()<=1) return v.size();
		int index = 1;
		for (int i = 1; i < v.size(); ++i)
			if (v[index - 1] != v[i])
				v[index++] = v[i];
		return index;
	}
	//Ԫ������ظ�����
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
	int Duplicate_twice(vector<int>& v) {
		if (v.size() <= 2) return v.size();
		int index = 2;
		for (int i = 2; i < v.size(); ++i)
			if (v[i] != v[index - 2])
				v[index++] = v[i];
		return index;
	}
	//ͨʽ������k���ظ�
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
	//������ת�����в���Ԫ�أ����ַ�
	//ʱ�临�Ӷ�O(logN)���ռ临�Ӷ�O(1)
	//û���ظ�Ԫ��
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
	//���ظ�Ԫ�أ����Ҿ�ֻҪ�ж�target�Ƿ���ڼ���
	//�߼���������һ���ģ�Ψһ�Ĳ�ͬ�ǵ�A[m]>=A[l]ʱ����ô[l,m]Ϊ�������еļ���Ͳ��ܳ����ˣ�����[1,2,1,1,1]
	//ֻ�轫���������������if(A[m]>A[l])ʱ��[l,m]���ǵ������У�if(A[m]==A[l])������[1,2,1,1,1]����ֻ�ܱ�ʾA[l]������target��l++��
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
	//���������������飬�ҳ����Ǻϲ������λ����Ҫ��ʱ�临�Ӷ�Ϊlog(M+N)
	//���������һ����������Ϊ--�ҳ��������������е�kС����
	//����һ����һ��������count����emerge��������ķ�ʽ����������ָ��PA��PB�ֱ�ָ���������飬���*PA<*PB����PA++,count++��
	//���*PA>*PB����PB++,count++��ֱ��count==k���������count=kʱ��Ӧ��*PA��*PB��ʱ�临�Ӷ�ΪO(m+n)���ռ临�Ӷ�O(1)��
	//������������α���ַ�����������Ĵ�Сm,n������k/2��ͨ�����Ƚ�A[k/2]��B[k/2]���Լ����ų�����Ԫ�أ�����A[k/2]==B[k/2]��
	//���ҵ��˵�kС��ֵ��A[k/2]<B[k/2]����A[0]-A[k/2]�����϶�С�ڵ�kС���Ǹ�������֮B[0]-B[k/2]С���Ǹ�target�������С��Χ��
	class Solution1 {
	public:
		double Mid_Num(vector<int>& a, vector<int>& b) {//����λ��
			int m = a.size();
			int n = b.size();
			int total = n + m;
			if (total & 0x1)
				return find_kth(a.begin(), m, b.begin(), n, total / 2 + 1)*1.0;
			else
				return (find_kth(a.begin(), m, b.begin(), n, total / 2 + 1) + find_kth(a.begin(), m, b.begin(), n, total / 2)) / 2.0;
		}
	private:
		int find_kth(vector<int>::iterator A, int m, vector<int>::iterator B, int n, int k) {//�ҵ�KС����
			//������m<=n
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
	//�������������ҵ�������������У��������ĳ��ȣ�Ҫ��ʱ�临�Ӷ�O(N)
	//����[100,6,2,99,5,3,4,1,98]���������������Ϊ[1,2,3,4,5,6]
	//����������hash������O(N)��,��������Ѱ������Ԫ�أ�O(N)��,��ΪO(N)
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
	//�ҵ������к͵���sum��������
	//��hash�����н���ֵ���±���ӳ�䣬�ٱ���Ѱ��Ŀ�����
	//ʱ�临�Ӷ�ΪO(N)
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
	//ͬ��ɵ�3sum=0���룬ʱ�临�Ӷ�ΪO(N)
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
	//closest3Sum������ƽ�ĳ��targetֵ���������ĺ�
	//ʱ�临�Ӷ�O(N**2)
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
	//ʱ�临�Ӷ�ΪO(N**2)
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
	//�õ�����������Դ����һ������
	//ʱ�临�Ӷ�O(N)���ռ临�Ӷ�O(1)
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
	//More Advanced Solution:���ر����
	//���ؽ���(������ǵڼ�������)�Ϳ��ؽ���(��k�����е����Ƕ���)
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
			if (k > f[n - 1] * n) return vector<int>{-1};//������Χ�������򷵻�-1
			--k;//��k-1������С����
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
	//�ж��Ƿ�����Ч�������ո�����'.'����ʾ��
	//����ֻҪ���9��9��9���Ź��񣬶�����ܸ��ӣ����Ը��Ӷȿ��Ժ���
	class solution4 {
	public:
		bool isValidSudoku(vector<vector<char>>& v) {
			bool used[9];
			for (int i = 0; i < 9; ++i) {
				fill(used, used + 9, false);
				for (int j = 0; j < 9; ++j) 
					if (!check(v[i][j], used))//�����
						return false;
				fill(used, used + 9, false);
				for (int j = 0; j < 9; ++j)
					if (!check(v[j][i], used))//�����
						return false;
			}
			for (int r = 0; r < 3; ++r) //���Ź���
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
	//ֱ��ͼ��ˮ����
	//ʱ�临�Ӷ�O(N)���ռ临�Ӷ�O(1)
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
	//������ת����
	//�ѷ���˳ʱ����ת90�ȣ��ȶԽ��߷�ת�������ž���Ĵ�ֱ�������ҷ�ת
	//�ѷ�����ʱ����ת90�ȣ�Ҳ���ȶԽ��߷�ת�����ǵڶ���������ˮƽ�������·�ת
	void rotateSquareI(vector<vector<int>>& v) {
		int n = v[0].size();
		//�Խ��߷�ת
		for (int i = 0; i < n; ++i)
			for (int j = i + 1; j < n; ++j)
				swap(v[i][j], v[j][i]);
		//˳ʱ����ת90�ȣ����ҷ�ת
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n / 2; ++j)
				swap(v[i][j], v[i][n - i - 1]);
	}
	void rotateSquareII(vector<vector<int>>& v) {
		int n = v[0].size();
		//�Խ��߷�ת
		for (int i = 0; i < n; ++i)
			for (int j = i + 1; j < n; ++j)
				swap(v[i][j], v[j][i]);
		//��ʱ����ת90�ȣ����·�ת
		for (int i = 0; i < n / 2; ++i)
			for (int j = 0; j < n; ++j)
				swap(v[i][j], v[n - 1 - i][j]);
	}
	//������һ����
	vector<int>& plusOne(vector<int> v) {
		int c = 1;//��λ
		for (auto i = v.rbegin(); i != v.rend(); ++i) {
			*i += c;
			c = *i / 10;
			*i %= 10;
		}
		if (c > 0) v.insert(v.begin(), c);
		return v;
	}
	//쳲�����¥��,f(n)=f(n-1)+f(n-2)
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
	//������
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
	//������0�ĺ�������
	//ʱ�临�Ӷ�O(m*n)���ռ临�Ӷ�(m+n)
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
	//�����ռ�����
	//Solution :���õ�һ�к͵�һ��
	//ʱ�临�Ӷ�O(m*n)���ռ临�Ӷ�O(1)
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
	//ʱ�临�Ӷ�O(N),�ռ临�Ӷ�O(1)
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
	//Find Single Number,���ظ�2�ε������ҵ�Ψһ����һ�ε���
	//��򣬳���ż���ζ���������
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
	int FindSingleNumberr(vector<int>& v) {
		size_t ret = 0;
		for (auto i : v)
			ret ^= i;
		return ret;
	}
	//Find Single Number,���ظ�3�ε������ҵ�Ψһ����һ�ε���
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
	struct ListNode {//��������ڵ�
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
	//Note : Given m, n satisfy the following condition : 1 �� m �� n �� length of list.
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
	ListNode* ReverseLinkedListII(ListNode* head, int m, int n) {
		if (head == nullptr || m == n)
			return head;
		ListNode* prev = head;
		for (int i = 1; i < m - 1; ++i)
			prev = prev->next;
		ListNode* head2 = prev;
		prev = head2->next;
		ListNode* cur = prev->next;

		for (int i = m+1; i <= n; ++i) {//ͷ�巨
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
			if (cur == nullptr) break;
			if (!cur->next || cur->val != cur->next->val) {
				prev->next = cur;
				prev = cur;
				cur = cur->next;
			}
		}
		prev->next = cur;
		return bg.next;
	}
	//Rotate Linked List
	//Given a list, rotate the list to the right by k places, where k is non-negative.
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
		cur->next = head;//��β�����ջ�
		for (int i = 0; i < span; ++i)
			cur = cur->next;
		head = cur->next;
		cur->next = nullptr;//�ϻ�
		return head;

	}
	//Remove the Nth from End of Linked List
	//For example, Given linked list: 1->2->3->4->5, and n = 2.
	//After removing the second node from the end, the linked list becomes 1->2->3->5.
	// Given n will always be valid. Try to do this in one pass.
	ListNode* RemoveNthFromEnd(ListNode* head, int n) {
		ListNode hret(-1);//�������νڵ㣬Ӧ�Կ��ܴ��ڵ�ɾ��ͷ���
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
		//prevָ�������ǰһ���ڵ㣬[begin,end]�Ǵ�reverse������.
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
			return prev;//���prev�βκ�ʵ��prev������ͬһ��ָ�룬����Ҫ����ʵ�ε�prev���ǵ÷��ظ���
		}
	};
	//Copy Linked List with random pointer
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
				if (cur->random)//ע����cur->random->nextʱ���ȵ��ж�cur->random����Ч��
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
	bool LinkedListCycle(ListNode* head, ListNode** pos = nullptr) {
		ListNode* slow = head;
		ListNode* fast = head;
		while (fast && fast->next) {
			slow = slow->next;
			fast = fast->next->next;
			if (slow == fast) {
				*pos = fast;
				return true;
			}
		}
		return false;
	}
	//Linked List Cycle II,Locate the begining node of cycle.
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)	
	ListNode* LinkedListCycleII(ListNode* head) {
		ListNode* pos;
		if (!LinkedListCycle(head, &pos)) return nullptr;
		ListNode* cur = pos;
		int loopSize = 1;
		while (cur->next != pos) {
			++loopSize;
			cur = cur->next;
		}
		ListNode* fast = head;
		for (int i = 0; i < loopSize - 1; ++i)
			fast = fast->next;
		pos = head;
		while (fast->next != pos) {
			fast = fast->next;
			pos = pos->next;
		}
		return pos;
	}
	//Reorder Linked List
	//For example, Given {1,2,3,4,5,6}, reorder it to {1,6,2,5,3,4}.
	//You must do this in-place without altering the nodes�� values.
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
	//LRU���棬��ʡ�ڴ��ͬʱ��߶��ļ��ķ����ٶ�
	class LRUCache {
	private:
		struct CacheNode
		{
			int key;//�������κ����͵�������Ϊ�˼򻯣��˴�Ϊint
			int val;//����Ϊ�κ����͵�class��struct��Ϊ�˼򻯣��˴�Ϊint
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
	//partion��Сд�ַ���С����ǰ����ں������λ�ò��䣬�����ռ�
	//���ֹ��ɣ�ɨ���ַ����������С�ĵ�ǰ���򽻻���ɨ��������ڴ�д�ַ�������
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
		//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
	//��һ���������
	//ʱ�临�Ӷ�O(m*n)���ռ临�Ӷ�O(1)
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
	//������KMPģʽƥ���㷨
	//ʱ�临�Ӷ�O(m+n)���ռ临�Ӷ�(m)
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
	//Implement atoi()�����Ϲ����뷵����
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
	//ʱ�临�Ӷ�O(n),�ռ临�Ӷ�O(1)
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
		string PreProcess(const string& s) {//���ż������
			string t = "$#";
			for (int i = 0; i < s.size(); ++i) {
				t += s[i];
				t += "#";
			}
			return t;
		}
	};
	//Regular Expression Match,Implement '*' and '.'
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
		//ʱ�临�Ӷ�O(m*n)���ռ临�Ӷ�O(1)ps:https://shmilyaw-hotmail-com.iteye.com/blog/2154716
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
		//ʱ�临�Ӷ�O(m*n)���ռ临�Ӷ�O(1)
		string LongestCommonPrefix(vector<string>& vstr) {
			for (int i = 0; i < vstr[0].size(); ++i)
				for (int j = 1; j < vstr.size(); ++j) 
					if (vstr[0][i] != vstr[j][i])
						return vstr[0].substr(0, i);//�±�Ϊiʱû�гɹ������Խ�ȡ���Ȳ���i+1������Ϊi
			return vstr[0];
		}
		//Valid Number
		//Valid Float:�����Python:
		//import re ; re.match(pattern,string); pattern: r"[-+]?(\d+\.?|\.\d+)\d*([e|E][-+]?\d+)?"
		//FloatҪ���ǵ����еĺϷ���������:+123 .123 - .123 .123E10 123.E10 �ͷǷ�����.E - 10 .e
		//C++������
		//����strtod(char* str,char** PtrEnd)
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
		//ʱ�临�ӶȺ�Integer�Ĵ�С�йأ��ռ临�Ӷ�O(1)
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
			const int value[13] = { 1000,900,500,400,100,90,50,40,10,9,5,4,1 };//�����������õ������޷�ʹ��ģ������int value[]={1}
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
		//ʱ�临�Ӷ�O(n2)���ռ临�Ӷ�O(1)
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
		//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
		//ʱ�临�Ӷ�O(Length of string)
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
	//�жϳɶԵ������Ƿ���Ч
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
	//�����Ч��������
	//Input:"))()([)][()[]])" Output:6
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
	//���Ľⷨ˼·�����������ջ https://www.cnblogs.com/ganganloveu/p/4148303.html#undefined
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
	//Evaluate Reverse Polish Notation "���沨�����ʽ"
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(n)
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
	//������ȱ��������� Level-Traversal
	//ʱ�临�Ӷ�O(n),�ռ临�Ӷ�O(n)
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
	//��һ��bool��¼�����һ��Ǵ��ҵ���ÿһ�������reverse()һ��
	//ʱ�临�Ӷ�O(n)���ռ临�ӵö�O(n)
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
			left_to_right = !left_to_right;//��һ�е�ȡ��˳��
			ret.push_back(level);
			swap(current, next);
		}
		return ret;
	}
	//Is Same Tree
	//ʱ�临�Ӷ�O(lgn)���ݹ��ռ临�Ӷ�O(1)��������ռ临�Ӷ�O(n)
	bool IsSameTree(TreeNode* p,TreeNode* q) {
		if (!p && !q) return true;
		if (!p || !q) return false;
		return p->val == q->val&&IsSameTree(p->left, q->left) && IsSameTree(p->right, q->right);
	}
	//������
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
		//Is Symmetric Tree ����������
		//�ݹ��
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
		//������
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
		//�ݹ���������߶�ʱ����ƽ���ж�
		//ʱ�临�Ӷ�O(logn)���ռ临�Ӷ�O(1)
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

			if (lHeight < 0 || rHeight < 0 || abs(lHeight - rHeight)>1) return -1;//�ж��Ƿ�Ϊƽ�������

			return max(lHeight, rHeight) + 1;//�����ϲ��������ĸ�
		}
		//������ͨ��������ȱ���ʱ�ۼƸ߶�
	};

	//Flatten Binary Tree To Linked List
	// https://www.cnblogs.com/grandyang/p/4293853.html �ⷨ��
	//������
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
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
				root = next;//ת����һ��Level
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
		//����N���ڵ�ֵ�������ܹ���ɶ���������������
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
	//���ַ�
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
//���ַ�������ָ��
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
	//����
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
	//����
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
	//�������������������
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
				if (cur->left) {   //��ڵ���ڲ�sum����
					vt.pop_back();
					sum += cur->val;
				}
				cur = cur->right;
			}
		}
		return vret;
	}
	//Maxium Sum Path,Return Maxium Vaule
	//������Maxium Substring Sum����֧�ʹ���0���ۼӣ�С��������
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
	//1->2->3 ���������123�����Ƶģ������и���Ҷ������ɵ��������,������0-9����
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
	//�鲢K������
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
	//Insertion Sort List ����
	//����ȡ��node����ͷɨ�����������ŵ� dummy(-1)->null���ʵ�λ��
	//ʱ�临�Ӷ�O(n**2)���ռ临�Ӷ�O(1)
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
	//ͨ���������ù鲢��˫�����ÿ���
	class LinkedListSorted {
		ListNode* mergeSorted(ListNode* head) {
			if (head == nullptr || head->next == nullptr) return head;
			//����ָ�������е�
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
	//�ҵ���һ��ȱʧ������,Ͱ��
	//ʱ�临�Ӷ�O(n)���ռ临�Ӷ�O(1)
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
					if (vbkt[i] <= 0 || vbkt[i] > vbkt.size() || vbkt[vbkt[i] - 1] == vbkt[i])//С�ڵ�����ʹ��ڱ߽��ֵ���ű��������ڸ�λ�õ���swap
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
	//Advanced Solution:One-Pass��ֻɨ��һ��
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
	//����������������������ö��ַ�
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
	//�������Array�У�target�����򷵻�����λ�ã����򷵻ظ�targetӦ��������λ��,With Requirement:Time Complex withIn O(logN)
	//ֱ����bound_lower����ʵ��bound_lower
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
	//ʱ�临�Ӷ�O(logN)
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
	//Subset,ȫ��û���ظ�
	//������
	//ѡ��ѡ->ʱ�临�Ӷ�O(2**N)���ռ临�Ӷ�O(1)
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
	//���ѣ�����û�м�֦����֮�ݹ����
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
	//More More Effective �����Ʒ�
	//ǰ����Ԫ������������int��λ��,{A,B,C,D}��6==0110��ʾ�Ӽ�{B,C}
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
	//��ȫ��Ԫ�����ظ�������{1,2,2}���Ӽ�,��������Ļ�����push�Ӽ�֮ǰ�����أ������STL�����ݽṹset()��Ȼȥ��
	vector<vector<int>> subSetWithBinary(vector<int>& v) {
		sort(v.begin(), v.end());
		vector<vector<int>> vret;
		vector<int> cur;
		unordered_set<string> set; //����д����Ҫvector<>ģ��֧��hash()���μ�https://blog.csdn.net/haluoluo211/article/details/82468061
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
	//OJ��վ�������
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
	//������˺�㷨
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
	//���ر����
	//�ж�������ǵڼ������У����ر���
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
		return k + 1;//��k������С�����У��������ǵ�k+1��
	}
	////�ҵ���k������
	//��������k-1��next_permutation()
	//Advaced:���ؽ���
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
	//������������е�Ԫ�������ظ����ⷨ�Ͳ��ظ���һ��
	//Combinations
	//Given two integers n and k, return all possible combinations of k numbers out of 1 ...n.
	//For example, If n = 4 and k = 2, a solution is :
	//[[2, 4],[3, 4],[2, 3],[1, 2],[1, 3],[1, 4],]
	//Solution I:Recursion
	//�����Ӽ��ĵݹ�����ƣ������������������ܼ���ֱ����DFS
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
	//[],[2],[1] [1,2]->��ȡ
	//[][3][2],[1] [1,3]->��ȡ,[2,3]->��ȡ
	//[1,2][1,3][2,3]
	//�����Ӽ��ĵ�������
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
		//�ڶ�������д������һ���ǻ���ջ����ʱ����cur��д�����ڶ����Ǳ�������&cur
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
	//������Ҫ����������·������
	//���������и��õ�Ч�ʣ����ռ俪��Ҳ����
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
	//�����������·���ļ���
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
			if (cur.size() > minLadder) //��֦
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
	//���Թ����·��
	int mazeShortestPath(vector<vector<int>>& vv, int startx, int starty, int endx, int endy) {
		if (vv.empty()) return -1;
		int m = vv.size();
		int n = vv[0].size();
		int** visit = new int*[m];//������vector<vector<int>> vv(m,vector<int>(n,0))���																					
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
namespace DFS
{
	//Surround Regions
	//���������򻮷�����,DFS
	//���󣺰ѱ�X��Χ��O��X���
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
	//���������Ƚ�ͻ������������
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
	//�����ַ�����ʹ���ÿһ���Ӵ����ǻ���
	//�Ӽ������⣬����ģ��subSet��Combination�ĵݹ�д��
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
	//Ϊ�˽�ʡ�ռ䣬���Բ�������ˢ��
	int uniquePathsAmount(int m, int n) {
		vector<int> f(n, 1);
		for (int i = 1; i < m; ++i)
			for (int j = 1; j < n; ++j)
				f[j] = f[j] + f[j - 1];
		return f[n - 1];
	}
	//Unique Paths II
	//��ͼ�м������ϰ���1��ʾ���ϰ������������ϵ㵽���µ��·�����������������ѵݹ飬Ҳ��������DP
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
					ss.pop_back();//����ĩβ��'.'
					vret.push_back(ss);
				}
				return;
			}
			//��������μ�֦
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
	//����ʹ���ظ�Ԫ��
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
	//������Ԫ���ظ�ʹ��
	//For example, In:1,2,7,6,1, target=8
	//Out:[1, 7],[2, 6],[1, 1, 6]
	//���������ƣ�ֻҪ��΢�Ķ���
	//�����ڵݹ��forѭ�������if (i > start && num[i] == num[i - 1]) continue; �������Է�ֹres�г����ظ��
	//Ȼ����ڵݹ����combinationSum2DFS����Ĳ�������i+1�������Ͳ����ظ�ʹ�������е�������
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
	//���ŵĵ���'.'��ʾ
	//��һά������dfs
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
	//ֱ��dfs
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
	//�����������������Խ��ƹ��ѵķ�ʽ�������ѵķ������������ʺ��ڹ������ȴ���޷����ܹ��ѵĿռ����Ķ�ʱ�乩������ʱ���õ�������������
	//������������dfs()�е�������ȴﵽ������maxDepth��ֵ���ҵ��˽�ռ�ʱ��ֹͣ������
	//α��������������
	/*for (int i = 0; i < finalDepth; ++i){//������finalDepth���������ռ���ȣ����������ö�ʱ��������������ȣ�����1��֮�ڣ�������������
		dfs(, 0, i, vret);
		if(timeOut())//����Ƿ�ʱ
		    break;
		}
	void dfs(,int start , int maxDepth, vector<int>& cur,vector<vector<int>>& vret) {
		if (start > maxDepth)
			return;
		if (�ҵ��˽�ռ�) {
			vret.push_back(��ռ�);
			return;
		}
		for (int i = start; i < ĳ��ֵ; ++i) {
			cur.push_back(ĳ���ڵ�);
			dfs(, i + 1, maxDepth, cur, vret);
			cur.pop_back();
		}
	}*/
}
namespace ReImplementFunction {
	//Re-Implement pow(x,n)
	//���ַ�
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
				return v*v*x; //����n=1Ҳ���뵽����߼��У�if (n == 1) return x;
			else
				return v*v;
		}
	};
	//Re-Implement sqrt(x)
	//���ַ�
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
	//̰���㷨���Ծֲ��������Ƶ�ȫ�����ŵķ����������Ƶ��Ľ����һ�������Ž⣬�����ǽ������Ž⣬�������Ž��������٣�����������Ȱ��š�
	//���ϸ������⡢���������⣬������NP��ȫ���⣬��������ѧ�����ϲ�û�п��ٵõ����Ž�ķ�����̰���㷨�����ʺϴ����������⣬�Եõ��������Ž�
	//Jump Game
	//�����������������i�����ľ�����a[i]���ж��ܲ����������ұ�
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
	//Best Time To Buy And Sell Stock��ֻ������һ��
	//ʱ�临�Ӷ�O(N)
	//�ҵ���͵ĵ����ߵĵ㣬����֤��͵�����ߵ��ǰ��
	int BuySellStock(const vector<int>& v) {
		//����Ǹ��ݼ����У�����-1
		int buy = INT_MAX;
		int profit = 0;
		for (int i = 0; i < v.size(); ++i) {
			buy = min(buy, v[i]);
			profit = max(profit, v[i] - buy);
		}
		return profit;
	}
	//Best Time To Buy And Sell Stock�������������״���
	//̰���㷨��������ļ۱Ƚ����ʱ��������������
	//ʱ�临�Ӷ�O(N)
	int BuySellStockII(const vector<int>& v) {
		int profit = 0;
		for (int i = 0; i < (int)v.size() - 1; ++i)//��Ϊ���޷��������������ȵý���ת�з��ţ��÷���ʽ����ȼ�������Ƿ�Ϊ��Ҳ��
			if (v[i] < v[i + 1])
				profit += v[i + 1] - v[i];
		return profit;
	}
	//LongestSubstring WithOut Repeation
	//ʱ�临�Ӷ�O(N),�ռ临�Ӷ�O(N)
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
	//ʱ�临�Ӷ�O(N)���ռ临�Ӷ�O(1)
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
	//�����ص����⣬������಻�ص������������
	//��಻�ص�������
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
	//����ص������
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
	//��Ȩ��������ȣ���ÿ�������ϰ�һ��Ȩ�أ����Ȩ֮������䳤�����ֵ
	//For example,ĳ�Ƶ���þ���ʽ��ס��ÿһ��������һ����Ԫ�飨��ʼ����סʱ�䣬ÿ����ã���������N�����꣬ѡ��ʹ�Ƶ�Ч�����ľ���
	//DP,��������ص�������ʱ����Ȩ��
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
		//����ʽ��̣������εĺϷ���
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
	//�������串��
	//�����ٵ����串��ĳһ����������
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
	//��������ص�
	//��start���򣬴�������ɨ�裬�����ڵ������ҳ������ص�����

	//���鰲�ţ���ɫ����
	//�൱���������ص�������
	typedef struct  Meet {
		int num;
		int time;//��ʼ�����ʱ��
		int sore;//��ʼΪ1,����Ϊ0
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
			//����ʽ���
			vector<Meet> vtime;
			for (int i = 0; i < vstart.size(); ++i) {
				vtime.push_back(Meet(i, vstart[i], 1));
				vtime.push_back(Meet(i, vend[i], 0));
			}
			sort(vtime.begin(), vtime.end(), cmp3);
			stack<int> stkroom; //׼������
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
	//��Ǯ����
	//�ܹ��ж����ֶһ�����
	//Solution 1:���ѣ���CombinationSum I һ�����ǲ���Ҫ�������ѽ���ģ�����������ָ�����������ᳬʱ
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
	//�����ܶһ�������ֽ��/öӲ��
	//̰��
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
	//�һ��̶�������Ӳ�ҵķ���
	//����+��֦
	void dfs(vector<int>& money, int start, vector<int>& cur, vector<vector<int>>& vret, int gap, int amount) {
		if (cur.size() == amount&&gap == 0) {
			vret.push_back(cur);
			return;
		}
		if (cur.size() > amount)//��֦
			return;
		for (int i = start; i < money.size(); ++i) {
			if (money[i] > gap) return;
			cur.push_back(money[i]);
			dfs(money, i, cur, vret, gap - money[i], amount);
			cur.pop_back();
		}
	}
	int getSpecificAmount(vector<int>& money, int target, vector<vector<int>>& vret, int amount) {//��target���㣬����amountöӲ��
		 //����ʽ��̣�������
		//...
		sort(money.begin(), money.end());
		vector<int> cur;
		dfs(money, 0, cur, vret, target, amount);
		if (vret.empty())
			return -1;
		return vret.size();
	}
	//����ճ������
	//��ʼ��һ��'A',�ṩ2�ֲ�����һ��Copy All,���ı���������A���Ƶ�ճ���壬����Paste��ճ�����ڵ�A׷�ӵ��ı���
	//��Ҫʹ�õ�n��'A'������Ҫ���м��β���
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
	//�ṩ4�ֲ�������һ��A��ȫѡ�����ƣ�ճ��
	//���Խ���N�β������������Ի�ö��ٸ�A
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
		//���ݳ�LSC�ܴ�
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
	//LIS �����������
	//�ҵ������п�����LCS�ķ�����ʱ�临�Ӷ�O(N**2)����ӹ���һ����������������ַ���
	//��ֻҪ�����г��ȣ���̰���㷨,ά��һ��������ջ�����ջ�Ĵ�С������������еĳ��ȣ�ʱ�临�Ӷ�O(N*logN)
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
	//�������Ź���lower_bound(x.begin(),x.end(),target)���ص�һ�����ڵ���target��λ�ã�upper_bound�Ƿ��ص�һ������target��λ�ã�
	//lower_bound��prevһ��ʹ�ÿ����ҵ�һ��С�ڵ�λ�ã�prev��upper_bound�����ҵ�һ��С�ڵ��ڵ�λ��
	
	//������������к�
	//ʱ�临�Ӷ�O(N)���ռ临�Ӷ�O(1)
	int MaxSumOfSubsequence(const vector<int>& v) {
		if (v.empty()) return -1;
		int maxSum = INT_MIN;
		int curmax = 0;
		for (int i = 0; i < v.size(); ++i) {
			curmax = max(curmax + v[i], v[i]);
			//��curmax>0ʱ��curmax=curmax+v[i];
			//��curmax<0ʱ��curmax=v[i];
			maxSum = max(maxSum, curmax);
		}
		return maxSum;
	}
	//������������л�
	//���Ǹ�������
	//ʱ�临�Ӷ�O(N)���ռ临�Ӷ�O(1)
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
	//����С�ļ�����ۣ��������ά���ֱ�Ϊ10*100��100*5��5*50�ľ���A��B��C��(A*B)*C�Ĵ�����7500�μ��㣬����A*(B*C)�Ĵ���75000�μ��㡣
	int MaxtrixChainMultiply(const vector<pair<int, int>>& vp) {//pair<int,int>�洢�˾��������
		if (vp.size() < 2) return 0;
		int n = vp.size();
		//Ԥ������������ȡ��һά����
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
	//Advanced,�ҵ�����Ļ��ַ���
	class MatrixChainMultiply {
	public:
		int MaxtrixChainMultiplyII(const vector<pair<int, int>>& vp) {//pair<int,int>�洢�˾��������
			if (vp.size() < 2) return 0;
			int n = vp.size();
			//Ԥ������������ȡ��һά����
			vector<int> v(n + 1, 0);
			for (int i = 0; i < n; ++i) {
				v[i] = vp[i].first;
				v[i + 1] = vp[i].second;
			}
			//DP
			vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
			vector<vector<int>> path(n + 1, vector<int>(n + 1, 0));//��¼���ֵ�
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
	//��������,01,���֣���ȫ
	//01
	class Bag01 {
	public:
		int bag01(const int cap, const vector<int>& volume, const vector<int>& value) {
			//Ԥ����
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
			//��ӡ��ѡ����Ʒ
			print(dp, vol, n, cap);
			return dp[n][cap];
		}
	private:
		void print(const vector<vector<int>>& dp, const vector<int>& vol, int i, int j) {//��ȫ�������dp[i][j]������
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
		//����ʽ��̣������κϷ���
		//...
		//Ԥ����
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
	//��ȫ
	int entireBag(const int cap, const vector<int>& volume, const vector<int>& value) {
		//����ʽ��̣������κϷ���
		//...
		//Ԥ����
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
	//��ש���ǣ�״̬ѹ��DP��
	//�� https://www.cnblogs.com/wuyuegb2312/p/3281264.html
	//״̬ѹ��+����+DP
	class CoverTileSolution {
	public:
		int coverTile(int m, int n) {
			if (n > m)//m��n��,��ȡС
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
			dfs(row, col + 1, n, dp, status, preRowStatus);//����
			if (col > n - 2 || (1 << col)&status || (1 << (col + 1))&status) return;//��֦
			dfs(row, col + 2, n, dp, status | (1 << (col + 1)) | (1 << col), preRowStatus);
		}
	};
	//���Էָ� Liner partision)
	//��һ���������˳�򲻱�ķ�ʽ�ֳ�K�ݣ�ʹK�����ݵ��������ȣ��;����ӽ���(��������һ�ݾ���С)
	//Solution I:����+��֦,��������ȵ���һ��
	//Solution II:DP

	//3�μ�ƻ��������ת��Ϊ3��1�صļ�ƻ��
	//ŷ����������������������̱պ�·�̣���Ҫ����ʽʱ�临�Ӷȣ�����ת��Ϊ˫��ŷ��������������⣬������O(N**2)ʱ�临�Ӷ���DP���.
	//������,����k��,(k>=1)���������˻�
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
	//һ����ֻ�е���3�ĸ����ﵽ���ʱ���Ĳ�˲Ż���󣬵������βʣ��1�Ļ�Ҫ�ٲ�1��3�Ӷ�ƴ��1��4
	int cutRopeII(const int length) {
		if (length == 0) return -1;
		if (length < 4) return length;
		if (length % 3 == 1)
			return pow(3, length / 3 - 1) * 4;
		else
			return pow(3, length / 3)*(length % 3 == 0 ? 1 : 2);
	}
	//paint fence
	//n����ʣ�k����ɫ�������������������Ϳ��ͬ����ɫ
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
	//�ж�s3�ǲ�����s2��s1��Ƕ��϶���,��s3.size()==s2.size()+s1.size()s2,
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
	//Solution I: ����,�����Ա������п��ܽ��
	//Solution II:������쳲�����¥��,DP���
	int decodeWays(const string& s) {
		if (s.empty()) return -1;//��μ���У����Բ��ü�����ַ��Ƿ�Ϊ��0������Ϊ�������⴮��"012356"��"123560"���Ways������0
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
	//�Ȼ������ӵ�DP���������ҹ��ɣ��Ƴ�ת�ƹ�ʽ
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
	//Solution I:����
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
	//�������п��ܵ����
	//Solution:����
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
	//ѡ�񡢲��롢ð�ݡ����١��鲢��ϣ�����ѡ�������������Ͱ���� ��10�־���������㷨���������㷨������������
	//�㷨�������е�׼�����ʱ�临�Ӷȡ��ռ临�Ӷȡ�������ȶ���
	//�����㷨��Ϊ�����㷨�ͷ������㷨����ʱ�临�Ӷ�ΪO(N)���㷨��Ϊ�����㷨
	//���㷨�Ĺ������ܲ��� https://blog.csdn.net/opooc/article/details/80994353
	//N!>x^n>...>3^n>2^n>N^x>...>N^3>N^2>NlogN>N>logN>1
	/*
	���������������ʱ�䵥λ����
	���ݹ�ģ|�������� �鲢���� ϣ������ ������
	1000�� |  0.75  1.22    1.77    3.57
	5000�� |  3.78  6.29    9.48    26.54
	1��    |  7.65  13.06   18.79   61.31
	*/
	//1.ϵͳ��sort ���������ִ�������ֵΪ��ֵ�ͣ���ʹ�ÿ��ţ�������ִ��Ļ��бȽ�������ʹ�ù鲢����
	//2.�鲢�Ϳ������ָ��죿
	//	���űȹ鲢����ĳ�����Ҫ�ͣ�����Ҫ�졣
	//3.Ϊʲô���й鲢�Ϳ��������أ�
	//	�ڱȽϵ�ʱ��ʹ�ñȽ�����ʱ��Ҫ׷��һ���ȶ��ԣ�ʹ�� �鲢���� ���Դ��ȶ��Ե�Ч����ʹ�ÿ��Ų��ܹ�ʵ���ȶ��Ե�Ч����
	//4.��Դ��ģ��ʱ�򣬵���������С�ڵ���60��ʱ��sort���� �����ڲ�ʹ�ò�������ķ�������һ����60����һ���Ĺ�ģ�����������ܵ͵�ʱ�򣬲�������ĳ�����͡�
	//5.��c��������һ�棬�ѹ鲢���򣬸ĳɷǵݹ飬�ǻ��ڹ����������ǡ�

	class ClassicalSortions {
	public:
		//����ð��
		void bubble(int arr[], int len) {
			for (int i = 0; i < len; ++i)
				for (int j = 1; j < len - i; ++j)
					if (arr[j - 1] > arr[j]) exchange(arr, j - 1, j);
		}
		//+����Ż�ð��
		//��һ��flag���ж�һ�£���ǰ�����Ƿ��Ѿ�����
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
		//+�ڲ��Ż�ð��
		//��һ��pos����¼��һ��ð���������������λ��
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
		//��������ĸĽ�����һ��ͻ��O(N**2)ʱ�临�Ӷȵ������㷨
		void shell(int arr[]);
		//���������
		void heap(int arr[], int len) {
			if (len < 2) return;
			for (int i = 0; i < len; ++i)
				buildHeap(arr, i);//����
			int tail = len - 1;
			exchange(arr, 0, tail);
			while (tail > 1) {
				keepHeap(arr, 0, tail--);//ά����
				exchange(arr, 0, tail);
			}
		}
		//���������㷨��������������ұȽϼ���ʱ�Ƚ���Ч
		void count(int arr[], int len) {
			if (len < 2) return;
			int maxNum = arr[0];
			for (int i = 1; i < len; ++i)
				if (arr[i] > maxNum) maxNum = arr[i];
			int* t = new int[maxNum + 1]();//�����������Ǹ���
			for (int i = 0; i < len; ++i) t[arr[i]]++;
			int j = 0;
			for(int i=0;i<=maxNum;++i)
				while (int count = t[i] > 0) {
					arr[j++] = i;
					t[i]--;
				}
			delete[]t;
		}
		//���ܲ���Ͱ���򣬵�ȴ��Ͱ������ʺϺ����������򣬻��ʺϷ���ֵ��ֵ����
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
		//���������һ�������ȶ����ķֲ���һ����Χ��ʱ��Ͱ����ȽϺ���
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
		//�����߽���
		void buildHeap(int arr[], int index) {
			while (arr[index] > arr[(index - 1) / 2]) {
				exchange(arr, (index - 1) / 2, index);
				index = (index - 1) / 2;
			}
		}
		//������ά���ѣ�ʱ�临�ӶȽ�Ϊlog(N)
		void keepHeap(int arr[], int index, int size) {
			while (index * 2 + 1 < size) {//��Ϊ����ȫ�������ṹ��ֻҪ��֤�ڵ��������Ч�ȿɣ�������Ч�Һ���һ����Ч
				int leftChild = index * 2 + 1;
				int maxChild = leftChild;
				if (leftChild + 1 < size&& arr[leftChild + 1]>arr[leftChild]) maxChild = leftChild + 1;
				if (arr[index] >= arr[maxChild]) return;
				exchange(arr, index, maxChild);
				index = maxChild;
			}
		}
	};
	//Top K ����
	//�Ӽ��������������ҳ�Top K ������
	//O(N*logK)
	class TopK {
	public:
		template<typename T> vector<T> topK(T arr[], int len, int k) {
			//����k�ѣ�����С�����������ѣ���֮����С��
			//������轨��С��
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
namespace OfferGot {
	//��ָoffer�ڶ��棬ϸ����
	//˳ʱ���ӡ����
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
			//�������Ҵ�ӡ
			for (int j = start; j <= endY; ++j)
				cout << vv[start][j] << "\t";
			//�������´�ӡ
			if (endX > start) {
				for (int i = start + 1; i <= endX; ++i)
					cout << vv[i][endY] << "\t";
			}
			//���������ӡ
			if (endX > start&& start < endY) {
				for (int j = endY - 1; j >= start; --j)
					cout << vv[endX][j] << "\t";
			}
			//�������ϴ�ӡ
			if (start < endY&&start < endX - 1) {
				for (int i = endX - 1; i > start; --i)
					cout << vv[i][start] << "\t";
			}
		}
	};
	//��������ĵ�һ�������ڵ�
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
	//��������˫��������һ������������ת��Ϊ˫���������ö���ռ�
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
	//����������һ���ڵ�
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
	//˫����ʵ��ջ
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
	//˫ջʵ�ֶ���
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
	//��ӡ��1��Nλ������
	//Solution I:���ַ���ģ���һ����
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

	//Solution II:�ݹ鹹��ȫ����
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

	//�������Ӽ�������
	//˫����˫����һ��һ��
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
	//�������ĳ˷�����
	class MultiBigNumbers {
	public:
		string MultiBigNums(string& s1, string& s2) {
			if (s1 == "" || s2 == "") return "empty Integer";
			if (!checkInput(s1) || !checkInput(s2)) return "invalid Integer";
			bool isPlus = true;
			if (isMinus(s1, s2)) isPlus = false;
			if (s1.size() > s2.size()) {
				string s = s1;
				s1 = s2;
				s2 = s;
			}
			s1 = (s1[0] == '-' || s1[0] == '+') ? string(next(s1.begin()), s1.end()) : s1;
			s2 = (s2[0] == '-' || s2[0] == '+') ? string(next(s2.begin()), s2.end()) : s2;
			int len1 = s1.size();
			int len2 = s2.size();
			int k = len1 + len2;
			vector<unsigned int> c(k + 1, 0);
			for (int i = 0; i < len1; ++i) {
				int carry = 0;
				for (int j = 0; j < len2; ++j) {
					int sum = (s1[len1 - 1 - i] - '0') * (s2[len2 - 1 - j] - '0') + c[k - i - j] + carry;
					carry = sum / 10;
					c[k - i - j] = sum % 10;
				}
				if (carry > 0) c[k - i - len2] = carry;
			}
			string sret;
			for (auto i : c)
				sret.push_back(i + '0');
			if (!isPlus) sret.insert(sret.begin(), '-');
			return sret;
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

		bool isMinus(string& s1, string& s2) {
			if (s1[0] != '-'&&s2[0] != '-') return false;
			if (s1[0] == '-'&&s2[0] == '-') return false;
			return true;
		}
	};
	//O(1)ʱ��ɾ������ڵ�
	//�������Ƿ�
	void deleteListNode(ListNode* head, ListNode* target) {
		if (head == nullptr || target == nullptr) return ;
		if (target->next) { //����β�ڵ�
			ListNode* next = target->next;
			target->val = next->val;
			target->next = next->next;
			delete next;
			next = nullptr;
		}
		else if (target==head) {//�����ڵ�
			head = nullptr;
			delete target;
			target = nullptr;
		}
		else {//����ڵ��β�ڵ�
			ListNode* cur = head;
			while (cur->next != target) cur = cur->next;
			cur->next = nullptr;
			delete target;
			target = nullptr;
		}
	}
	//����ƥ����ʽ(.��*)
	//����״̬��

	bool matchCore(const char* s, const char* p) {
		if (*p == '\0') return *s == '\0';
		if (*(p + 1) == '*') 
			if (*p == *s || (*p == '.'&&*s != '\0'))
				return matchCore(s + 1, p + 2) || matchCore(s + 1, p) || matchCore(s, p + 2);//a*������3��״̬��*�������ã���ǰһ���ַ��ظ�����ǰһ���ַ�ע��
			else 
				return matchCore(s + 1, p + 2);//�����ǰ�ַ���ƥ�䣬����һ��ģʽ���ַ���*����ע����ǰ�ַ�
		if (*p == *s || (*p == '.'&&*s != '\0')) return matchCore(s + 1, p + 1);
		return false;
	}
	//����ƥ����ʽ(*��?)
	//�˴�?��ʾ���ⵥ���ַ���*��ʾ���ⳤ�ȵ��ַ���
	bool matchCoreII(const char* s, const char* p) {
		if (*p == '*') {
			while (*p == '*') ++p;
			if (*p == '\0') return true;
			while (*s != '\0' && !matchCoreII(s, p)) ++s;
			return *s != '\0';
		}
		else if (*p == '\0' || *s == '\0') return *p == *s;
		else if (*p == *s || (*s != '\0'&&*p == '?')) return matchCoreII(++s, ++p);
		return false;
	}
	//�жϱ�ʾ��ֵ���ַ����Ƿ���Ч
	bool isValidNum(const char* s) {
		if (s == nullptr) return false;
		if (*s == '+' || *s == '-') ++s;
		if (*s == '\0') return false;
		if (*s == '.') {
			++s;
			if (*s<'0' || *s>'9') return false;
			while (*s >= '0'&&*s <= '9') ++s;
		}
		else {
			if (*s<'0' || *s>'9') return false;
			while (*s >= '0'&&*s <= '9') ++s;
			if (*s == '.') ++s;
			while (*s >= '0'&&*s <= '9') ++s;
		}
		if (*s == '\0') return true;
		if (*s == 'e' || *s == 'E') {
			++s;
			if (*s == '+' || *s == '-') ++s;
			if (*s<'0' || *s>'9') return false;
			while (*s >= '0'&&*s <= '9') ++s;
		}
		if (*s == '\0') return true;
		else
			return false;
	}
	//�ó����ռ�ʹ��������λ��ż��ǰ�棬���λ�ò���
	//��ʱ�任�ռ�
	void oddtoFront(vector<int>& v) {
		if (v.size() == 0) return;
		int n = 0;
		for (auto i : v)
			if (i % 2 == 1) ++n;  //Flag
		for (int j = 0; j < n; ++j)
			for (int i = 1; i < v.size(); ++i)
				if (v[i] % 2 == 1) {//�˴�Ӧ��Flag���ı��ʽ x%2 == 1һ��
					int tmp = v[i];
					v[i] = v[i - 1];
					v[i - 1] = tmp;
				}
	}
	//����ɿ���չ����
	void featureToFront(vector<int>& v, bool(*func)(int)) {
		if (v.size() == 0) return;
		int n = 0;
		for (auto i : v)
			if (func(i)) ++n;
		for(int i=0;i<n;++i)
			for(int j=1;j<v.size();++j)
				if (func(v[j])) {
					int tmp = v[j];
					v[j] = v[j - 1];
					v[j - 1] = tmp;
				}
	}
	//������չ����
	//������ǰ��
	bool isOdd(int num) {
		return num % 2 == 1;
	}
	//����featureToFront(v, isOdd);

	//������ݽṹ--��ջ��ʹ֮���к���push,pop,�����ӻ����Сֵ�ĺ��� min����3��������ʱ�临�Ӷ�ΪO(1)
	//�ÿռ任ʱ�䷽����ÿ��pushʱ��������ջѹ�뵱ǰ��Сֵ��ÿ��popʱ������ջҲpop��min�����򷵻ظ���ջջ��Ԫ��
	template<typename T>
	class StackLike {
	private:
		stack<T> stk;
		stack<T> stkMin;
	public:
		T min() {
			assert(!stk.empty() && !stkMin.empty());
			return stkMin.top();
		}
		void push(T val) {
			stk.push(T);
			if (stkMin.empty()|| T < stkMin.top())
				stkMin.push(T);
			else
				stkMin.push(stkMin.top());
		}
		void pop() {
			if (stk.empty()|| stkMin.empty()) return;
			stk.pop(); stkMin.pop();
		}
	};
	//����һ��˳����ջ�����п��ܵĳ�ջ������������
	//Solution : �����󣬿������� (2*n)!/(n+1)!/n!
	//�ж�һ�������Ƿ�Ϊ��һ��������ջ�Ŀ��ܵĳ�ջ����
	bool isPopSequence(vector<int>& va, vector<int>& vb) {
		if (va.empty() || vb.empty()) return false;
		if (va.size() != vb.size()) return false;
		stack<int> stkHelper;
		for (auto i = vb.rbegin(); i != vb.rend(); ++i)
			stkHelper.push(*i);
		stack<int> stk;
		for (auto i : va)
			if (i == stkHelper.top()) stkHelper.pop();
			else 
				stk.push(i);
		while (!stk.empty() && !stkHelper.empty() && stk.top() == stkHelper.top())
			stk.pop(), stkHelper.pop();
		return stk.empty();
	}
	//�����һ��˳����ջ���е����п��ܵĳ�ջ����
	//Solution I : ����ջ����ȫ����next_permutation()��������isPopSequence�ų������ܵ����
	//Solution II:ģ��ѹջ�͵�ջ���ݹ�
	//�ҵ�ģ��--��̫���ţ�Ҫȥ��
	void pushOrnot(vector<int>& v, int start, vector<int> cur, stack<int>& stk, unordered_set<string>& vret) {
		if (cur.size() == v.size()) {
			string s;
			for (auto i : cur) s += i + '0';
			vret.insert(s);
			return;
		}
		if (!stk.empty()) {
			for (int i = 0; i <= stk.size(); ++i) {
				stack<int> bak = stk;
				for (int j = 0; j < i; ++j) {
					cur.push_back(stk.top());
					stk.pop();
				}
				if (cur.size() == v.size()) pushOrnot(v, start, cur, stk, vret);
				if (start != v.size()) {
					cur.push_back(v[start]);
					pushOrnot(v, start + 1, cur, stk, vret);
					cur.pop_back();
					stk.push(v[start]);
					pushOrnot(v, start + 1, cur, stk, vret);
					if (!stk.empty()) stk.pop();
				}
				for (int j = 0; j < i; ++j) cur.pop_back();
				stk = bak;
			}
		}
		else {
			if (start != v.size()) {
				cur.push_back(v[start]);
				pushOrnot(v, start + 1, cur, stk, vret);
				cur.pop_back();
				stk.push(v[start]);
				pushOrnot(v, start + 1, cur, stk, vret);
				if (!stk.empty()) stk.pop();
			}
		}
	}
	//�����ҵĸ�����Щ--
	void pushOrnotII(vector<int>&v, int start, int N, vector<int>& cur, stack<int>& stk, vector<vector<int>>& vret) {
		if (start == N) {
			if (!stk.empty()) {
				int top = stk.top();
				stk.pop();
				cur.push_back(top);
				pushOrnotII(v, start, N, cur, stk, vret);//��Ȼ��start=N,ʹ֮������ջ
				cur.pop_back();
				stk.push(top);
			}
			else
				vret.push_back(cur);
		}
		else {
			stk.push(v[start]);
			pushOrnotII(v, start + 1, N, cur, stk, vret);
			stk.pop();
			if (!stk.empty()) {
				int top = stk.top();
				stk.pop();
				cur.push_back(top);
				pushOrnotII(v, start, N, cur, stk, vret);
				cur.pop_back();
				stk.push(top);
			}
		}
	}
	//�ж��Ƿ�Ϊ�����������ϵĺ�����������
	bool isValidSearchTree(const vector<int>& v, int start, int end);
	bool isPostTraversal(const vector<int>& v) {
		if (v.empty()) return false;
		return isValidSearchTree(v, 0, v.size() - 1);
	}
	bool isValidSearchTree(const vector<int>& v, int start, int end) {
		if (start >= end) return true;
	
		int rootVal = v[end];
		int i = end-1;
		while (i >= start&&v[i] > rootVal) --i;
		
		int j = i;
		while (j >= start&& v[j] < rootVal) --j;
		if (j >= start) return false;
		return isValidSearchTree(v, start, i) && isValidSearchTree(v, i+1, end - 1);
	}
	//���л��ͷ����л� ������
	//ǰ����������л�Ϊ�������Ա�ʾ��ͨ�����Ա�ʾ�������ع���������ʵ�ַ����л�
	class SerialBinTree {
		//���������ʹ���
		//stringstream ss;
		//ostream os(ss.rdbuf());
		//istream is(ss.rdbuf());
		//vector<int> v = { 5,2,3,6,1,8 };
		//TreeNode* root = nullptr;
		//for (auto i : v) root = buildTree(root, i);
		//SerialBinTree obj;
		//obj.serial(root, os);
		//cout << ss.str() << endl;
		//TreeNode* reroot = nullptr;
		//reroot = obj.deSerial(reroot, is);
		//printTree(reroot);
	public:
		void serial(TreeNode* root, ostream& stream) {
			if (root == nullptr) {
				stream << '$';
				return;
			}
			stream << root->value;
			serial(root->left, stream);
			serial(root->right, stream);
		}
		TreeNode* deSerial(TreeNode* root, istream& stream) {
			int num;
			if (readChar(num, stream)) {
				root = new TreeNode(num);
				root->left = deSerial(root->left, stream);
				root->right = deSerial(root->right, stream);
			}
			return root;
		}
	private:
		bool readChar(int& num, istream& stream) {
			char c;
			stream.get(c);
			if (c<'0' || c>'9') return false;
			num = c - '0';
			return true;
		}
	};
	//ȫ����
	//�ݹ��next_permutation()���˴��õݹ�
	class allPermutation {
	public:
		void permutation(string& s) {
			if (s.empty()) return;
			return recurPermutation(s, &s[0]);
		}
	private:
		void recurPermutation(string& s, char* begin) {
			if (begin == '\0') {
				cout << s << endl;
				return;
			}
			for (char* p = begin; *p != '\0'; ++p) {
				swap(*begin, *p);
				recurPermutation(s, begin+1);
				swap(*begin, *p);
			}
		}
	};
	//�ҳ������г��ִ�������һ�����
	//������ֻ��һ����������
	//Solution I:�����鲻���޸�ʱ�����ݸ����ĸ��������������������ܸ������ü�һ��һ����ʱ�临�Ӷ�O(N)
	bool checkMoreHalf(const vector<int>& v,int num) {
		int count = 0;
		for (auto i : v)
			if (i == num)
				++count;
		return count > v.size() / 2;
	}
	int findApearMoreHalf(const vector<int>& v) {
		assert(!v.empty());
		int res;
		int count = 0;
		for (int i = 0; i < v.size(); ++i) {
			if (count == 0) {
				res = v[i];
				count = 1;
			}
			else if (res == v[i]) count++;
			else --count;
		}
		assert(checkMoreHalf(v, res));
		return res;
	}
	//Solution II:�����޸�ԭ���飬���൱���������λ��������N/2�����
	//Solution STL:nth_element()Ѱ����λ��
	//Ҳ���� Quick Select ��ʱ�临�Ӷ�ΪO(N),��ѡ��ÿ��ѡһ���֣��ӵ���һ���֣�������O(N),����ÿ���ӵ�һ��.T(N) = n + n / 2 + n / 4 + n / 8 + n / 2 ^ k = n*(1 - 2 ^ -k) / (1 - 2 ^ -1) = 2N
	class QuickSelect {
	public:
		int findApearMoreHalf(vector<int>& v) {
			assert(!v.empty());
			return quickSelect(v,v.size()/2);
		}
		int quickSelect(vector<int>& v,int k) {
			int start = 0;
			int end = v.size() - 1;
			int pos = partition(v,start,end);
			while (pos != k) {
				if (pos < k)
					pos = partition(v, pos + 1, end);
				else
					pos = partition(v, start, pos - 1);
			}
			assert(checkMoreHalf(v, v[pos]));
			return v[pos];
		}
	private:
		int partition(vector<int>& v,int start,int end) {

		}
	};
	//�ҳ������г��ִ�������1/3����

	//��С��K����
	//Solution I :����������޸ģ������partion��˼�룬��Quick Select �ҵ���K�����������������߼�����С��K������ʱ�临�Ӷ�ΪO(N)
	//Solution II:�����鲻���޸ģ���ά��һK���ѣ�ʱ�临�Ӷ�ΪN*lgK���ǳ��ʺϺ������ݴ�����Ϊ�ڴ治���������ζ���

	//�������е���λ��
	//Solution I:����+Quick Select ���ʱ�临�Ӷ�O(1),����ʱ�临�Ӷ�O(n)
	//Solution II: ����+���� ���ʱ�临�Ӷ�O(n)������ʱ�临�Ӷ�O(1)
	//Solution III:����+���� ���ʱ�临�Ӷ�O(n)������ʱ�临�Ӷ�O(1)
	//Solution IV: ���С�ѷ� ���ʱ�临�Ӷ�O(logN)������ʱ�临�Ӷ�O(1)
	//���С�ѷ���ָ������������С�ѣ�ż���������ѣ�����Ϊ����ʱ����С�ѶѶ�Ԫ�أ�ż��ʱ���ش�Ѷ���С�Ѷ���ƽ����
	template<typename T>
	class DynamicMiddleNum {
	public:
		void add(T num) {
			int size = min.size() + max.size();
			if (size & 1 == 0) {
				if (max.size() > 0 && num < max[0]) {
					max.push_back(num);
					push_heap(max.begin(), max.end(), less<T>());

					num = max[0];

					pop_heap(max.begin(), max.end(), less<T>());
					max.pop_back();
				}
				min.push_back(num);
				push_heap(min.begin(), min.end(), greater<T>());
			}
			else {
				if (min.size() > 0 && num > min[0]) {
					min.push_back(num);
					push_heap(min.begin(), min.end(), greater<T>());

					num = min[0];

					pop_heap(min.begin(), min.end(), greater<T>());
					min.pop_back();
				}
				max.push_back(num);
				push_heap(max.begin(), max.end(), less<T>());
			}
		}
		T getMidNum() {
			int size = min.size() + max.size();
			assert(size > 0);
			if (size & 1)
				return min[0];
			else
				return (max[0] + min[0]) / 2;
		}
	private:
		vector<T> min;
		vector<T> max;
	};
	//��1-n�г�������1���ܸ���
	//Solution I:����ö�ٷ���ʱ�临�Ӷ�O(N*logN)
	//Solution II:��ѧ�۲죬���нⷨ��ʱ�临�Ӷ�O(logN)
	class Apear1Count {
	public:
		int count1(int n) {
			if (n <= 0) return 0;
			string s;
			while (n != 0) {
				s.insert(s.begin(), n % 10 + '0');
				n /= 10;
			}
			return permutateCount(s.c_str());
		}
	private:
		int permutateCount(const char* c) {
			if (!c || *c == '\0') return 0;

			int fstNum = *c - '0';
			int len = strlen(c);

			if (len == 1 && fstNum == 0) return 0;
			if (len == 1 && fstNum >= 1) return 1;

			int fistCounts;
			if (fstNum > 1) fistCounts = pow(10, len - 1);
			if (fstNum == 1) fistCounts = atoi(c + 1) + 1;

			int otherDigitBeOne = fstNum*(len - 1)*pow(10, len - 2);
			int recurvCounts = permutateCount(c + 1);
			return fistCounts + otherDigitBeOne + recurvCounts;
		}
	};
	//1-n��������1���ֵĸ���
	int getNthCharOfNumString(int n) {
		assert(n >= 0);
		if (n < 10) return n;
		if (n == 10) return 1;
		int sum = 1;
		int i = 0;
		while (sum < n) {
			++i;
			sum += i*(pow(10, i) - pow(10, i - 1));
		}
		if (sum == n) return 9;
		int dif = sum - n;
		int topNum = (int)pow(10, i) - 1;
		int base = i;
		while (dif > base) {
			dif -= base;
			--topNum;
		}
		int ret;
		while (topNum > 0 && dif>0) {
			ret = topNum % 10;
			topNum /= 10;
			--dif;
		}
		return ret;
	}
}