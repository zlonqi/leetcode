#include<fstream>
#include<iostream>
#include<string>
#include<algorithm>
#include<vector>
using namespace std;
//2017 Round C
namespace RoundC_2017{
	//Ambiguous Cipher
	void AmbiguousCipher() {
		ifstream infile;
		infile.open("A-large-practice.in");
		ofstream outfile;
		outfile.open("A-large-practice.out");


		cout << "Reading from the file" << endl;
		int line;
		infile >> line;
		for (int i = 0; i < line; ++i) {
			string s;
			infile >> s;
			int length = s.size();
			string t(length, '0');
			if (length % 2 == 0) {
				t[1] = s[0];
				for (int j = 3; j < length; j += 2)
					t[j] = (s[j - 1] - t[j - 2] + 26) % 26 + 'A';
				t[length - 2] = s[length - 1];
				for (int j = length - 4; j >= 0; j -= 2)
					t[j] = (s[j + 1] - t[j + 2] + 26) % 26 + 'A';
				outfile << "Case #" << i + 1 << ": " << t << endl;
			}
			else
				outfile << "Case #" << i + 1 << ": AMBIGUOUS" << endl;
		}
	}
	//Squared X
	void SquaredX() {
		ifstream infile;
		infile.open("B-small-practice.in");
		ofstream outfile;
		outfile.open("B-small-practice.out");


		cout << "Reading from the file" << endl;
		int loops;
		infile >> loops;

		for (int l = 0; l < loops; ++l) {
			int n;
			infile >> n;
			vector<string> vs;
			string s;
			for (int i = 0; i < n; ++i) {
				infile >> s;
				vs.emplace_back(s);
			}
			int twoX = 0, oneX = 0;
			vector<int> first;
			vector<int> second;
			for (auto i = 0; i < n; ++i) {
				int count = 0;
				vector<int> vcur;
				for (auto j = 0; j < n; ++j)
					if (vs[i][j] == 'X') {
						++count;
						vcur.emplace_back(j);
					}
				if (count == 1)
					++oneX;
				if (count == 2) {
					++twoX;
					first.emplace_back(vcur[0]);
					second.emplace_back(vcur[1]);
				}
			}

			bool isValid = true;
			if (oneX != 1 || twoX != n - 1) {
				outfile << "Case #" << l + 1 << ": IMPOSSIBLE" << endl;
				continue;
			}
			else {
				sort(first.begin(), first.end());
				sort(second.begin(), second.end());
				for (int i = 0; i < first.size(); i += 2)
					if (first[i] != first[i + 1] || second[i] != second[i + 1]) {
						outfile << "Case #" << l + 1 << ": IMPOSSIBLE" << endl;
						isValid = false;
						break;
					}
			}
			for (auto i : first)
				cout << i;
			cout << endl;
			for (auto j : second)
				cout << j;
			cout << endl;
			if (!isValid) continue;
			twoX = 0, oneX = 0;
			first.clear(); second.clear();
			for (auto j = 0; j < n; ++j) {
				int count = 0;
				vector<int> vcur;
				for (auto i = 0; i < n; ++i)
					if (vs[i][j] == 'X') {
						++count;
						vcur.emplace_back(i);
					}
				if (count == 1)
					++oneX;
				if (count == 2) {
					++twoX;
					first.emplace_back(vcur[0]);
					second.emplace_back(vcur[1]);
				}
			}

			if (oneX != 1 || twoX != n - 1) {
				outfile << "Case #" << l + 1 << ": IMPOSSIBLE" << endl;
				continue;
			}
			else {
				sort(first.begin(), first.end());
				sort(second.begin(), second.end());
				for (int i = 0; i < first.size(); i += 2)
					if (first[i] != first[i + 1] || second[i] != second[i + 1]) {
						outfile << "Case #" << l + 1 << ": IMPOSSIBLE" << endl;
						isValid = false;
						break;
					}
			}
			if (!isValid) continue;
			outfile << "Case #" << l + 1 << ": POSSIBLE" << endl;
		}
	}
}