#include <algorithm>
#include <unordered_map>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// uint16_t indices: (1 bit spin) (7 bits x) (8 bits y);
// spin: 0 up 1 down
// x, y: 64, 64 is origin (16448), assume will never reach near edges
#define ORIGIN_X 64
#define ORIGIN_Y 64
#define ORIGIN (ORIGIN_X*256 + ORIGIN_Y)
#define TO_SXY(s, x, y) (((s)*128 + (x) + ORIGIN_X)*256 + (y) + ORIGIN_Y)
#define TO_S(sxy) ((sxy) >> 15)
#define TO_X(sxy) (((sxy) >> 8 & 0x7f) - ORIGIN_X)
#define TO_Y(sxy) (((sxy) & 0xff) - ORIGIN_Y)

#define ONE_IF_EVEN(i) (1 - 2*((i) & 1))
#define ONE_IF_ODD(i) (2*((i) & 1) - 1)

#define MAX_OP 6

//#define tp -0.25

struct term { // 64 bytes
	double p; // prefactor
	uint16_t d[MAX_OP], c[MAX_OP], a[MAX_OP];
	int8_t nd, nca; // number of d operators, c/a operators
};

// kahan summing number
struct kdouble {
	double val, err;
	kdouble(double v = 0.0,  double e = 0.0) {val = v; err = e;}
	void operator = (const double& other) {
		val = other;
		err = 0.0;
	}
	kdouble operator + (const kdouble& other) {
		double tot_val = val + other.val;
		double tot_err = err + other.err;
		double new_val = tot_val + tot_err;
		double new_err = (tot_val - new_val) + tot_err;
		return kdouble(new_val, new_err);
	}
	kdouble operator * (const double& other) {
		return kdouble(val*other, err*other);
	}
	void operator += (const double& other) {
		double tot_err = err + other;
		double new_val = val + tot_err;
		err = (val - new_val) + tot_err;
		val = new_val;
	}
	void operator += (const kdouble& other) {
		double tot_val = val + other.val;
		double tot_err = err + other.err;
		val = tot_val + tot_err;
		err = (tot_val - val) + tot_err;
	}
	void operator -= (const double& other) {
		double tot_err = err - other;
		double new_val = val + tot_err;
		err = (val - new_val) + tot_err;
		val = new_val;
	}
	void operator -= (const kdouble& other) {
		double tot_val = val - other.val;
		double tot_err = err - other.err;
		val = tot_val + tot_err;
		err = (tot_val - val) + tot_err;
	}
	// operator double() const { return val + err; }
};

// array stuff: these are tiny arrays so O(n) performance is okay
// return index of x in a or -1 if not found
static inline int in_array(uint16_t n, const uint16_t *a, int na)
{
	for (int i = 0; i < na; i++) {
		if (a[i] == n) return i;
		if (a[i] > n) return -1;
	}
	return -1;
}

// insert n into a and return parity change of [n, old a] -> [new a]
static inline int finsert(uint16_t n, uint16_t *a, int na)
{
	for (int i = na-1; i >= 0; i--) {
		if (a[i] < n) {
			a[i+1] = n;
			return ONE_IF_ODD(i);
		}
		a[i+1] = a[i];
	}
	a[0] = n;
	return 1;
}

// remove a[i] and return parity change of [old a] -> [n, new a]
static inline int fremove(int i, uint16_t *a, int na)
{
	for (int j = i; j < na-1; j++)
		a[j] = a[j+1];
	a[na-1] = 0; // keeps unused space zero'd
	return ONE_IF_EVEN(i);
}

// remove a[i], insert n, and return parity change of
// [old a with a[i] replaced by n] -> [new a]
static inline int freplace(int i_old, uint16_t n, uint16_t *a, int na)
{
	int i = i_old;
	if (n >= a[i]) {
		for (; i < na-1; i++) {
			if (a[i+1] >= n) break;
			a[i] = a[i+1];
		}
	} else {
		for (; i > 0; i--) {
			if (a[i-1] <= n) break;
			a[i] = a[i-1];
		}
	}
	a[i] = n;
	return ONE_IF_EVEN(i - i_old);
}

static void term_print(const struct term *t)
{
	printf("%g (", t->p);
	for (int i = 0; i < t->nd; i++) {
		printf(i==t->nd-1?"(%d, %d, %d)":"(%d, %d, %d), ",
		       TO_S(t->d[i]), TO_X(t->d[i]), TO_Y(t->d[i]));
		if (t->nd == 1) printf(",");
	}
	printf(") (");
	for (int i = 0; i < t->nca; i++) {
		printf(i==t->nca-1?"(%d, %d, %d)":"(%d, %d, %d), ",
		       TO_S(t->c[i]), TO_X(t->c[i]), TO_Y(t->c[i]));
		if (t->nca == 1) printf(",");
	}
	printf(") (");
	for (int i = 0; i < t->nca; i++) {
		printf(i==t->nca-1?"(%d, %d, %d)":"(%d, %d, %d), ",
		       TO_S(t->a[i]), TO_X(t->a[i]), TO_Y(t->a[i]));
		if (t->nca == 1) printf(",");
	}
	printf(")\n");
}

// move terms so that a[0] or c[0] is at origin
static inline void term_shift(struct term *t, const int use_c)
{
	const uint16_t ref = t->nca > 0 ? (use_c == 0 ? t->a[0] : t->c[0]) \
	                                : t->nd > 0 ? t->d[0] : 0;
	if (ref == 0) return;
	const int16_t shift = ORIGIN - (ref & 0x7fff);
	if (shift == 0) return;
	for (int i = 0; i < t->nd; i++)
		t->d[i] += shift;
	for (int i = 0; i < t->nca; i++)
		t->c[i] += shift;
	for (int i = 0; i < t->nca; i++)
		t->a[i] += shift;
}

// flip spins
static void term_flip(struct term *t)
{
	int nup;
	uint16_t buf[MAX_OP];
	for (int i = 0; i < t->nd; i++)
		buf[i] = t->d[i];
	for (nup = 0; TO_S(buf[nup]) == 0 && nup < t->nd; nup++) ;
	for (int i = 0; i < t->nd; i++)
		t->d[i] = buf[(i+nup)%t->nd] ^ 0x8000;
	for (int i = 0; i < t->nca; i++)
		buf[i] = t->c[i];
	for (nup = 0; TO_S(buf[nup]) == 0 && nup < t->nca; nup++) ;
	for (int i = 0; i < t->nca; i++)
		t->c[i] = buf[(i+nup)%t->nca] ^ 0x8000;
	for (int i = 0; i < t->nca; i++)
		buf[i] = t->a[i];
	for (int i = 0; i < t->nca; i++)
		t->a[i] = buf[(i+nup)%t->nca] ^ 0x8000;
}

// flip and invert spins
static inline uint16_t flip_invert(const uint16_t n)
{
	return ((n&0x8000)^0x8000) ^ (2*ORIGIN - (n&0x7fff));
}
static void term_flip_invert(struct term *t)
{
	uint16_t buf[MAX_OP];
	for (int i = 0; i < t->nd; i++)
		buf[i] = t->d[i];
	for (int i = 0; i < t->nd; i++)
		t->d[i] = flip_invert(buf[t->nd-1 - i]);
	for (int i = 0; i < t->nca; i++)
		buf[i] = t->c[i];
	for (int i = 0; i < t->nca; i++)
		t->c[i] = flip_invert(buf[t->nca-1 - i]);
	for (int i = 0; i < t->nca; i++)
		buf[i] = t->a[i];
	for (int i = 0; i < t->nca; i++)
		t->a[i] = flip_invert(buf[t->nca-1 - i]);
}

// returns number of terms written to out
static int term_Kcomm(const struct term *t, struct term *out, const int16_t b, const double hop)
{
	int nterms = 0;
	for (int i = 0; i < t->nd; i++) { // d^dagger operators
		const uint16_t o = t->d[i];
		const uint16_t n = o + b;
		int i_n;
		if ((i_n = in_array(n, t->a, t->nca)) >= 0) {
			out[nterms] = *t;
			freplace(i, n, out[nterms].d, out[nterms].nd);
			out[nterms].p *= -hop*freplace(i_n, o, out[nterms].a, out[nterms].nca);
			term_shift(out+nterms, 0);
			nterms++;
		} else if (in_array(n, t->d, t->nd) == -1 && in_array(n, t->c, t->nca) == -1) {
			out[nterms] = *t;
			fremove(i, out[nterms].d, out[nterms].nd);
			out[nterms].nd--;
			int sign = ONE_IF_EVEN(out[nterms].nca);
			sign *= finsert(n, out[nterms].c, out[nterms].nca);
			sign *= finsert(o, out[nterms].a, out[nterms].nca);
			out[nterms].nca++;
			out[nterms].p *= hop*sign;
			term_shift(out+nterms, 0);
			nterms++;
		}
	}
	for (int i = 0; i < t->nca; i++) { // c
		const uint16_t n = t->c[i] + b;
		int i_n;
		if ((i_n = in_array(n, t->a, t->nca)) >= 0) {
			out[nterms] = *t;
			finsert(n, out[nterms].d, out[nterms].nd);
			out[nterms].nd++;
			int sign = ONE_IF_ODD(out[nterms].nca);
			sign *= fremove(i, out[nterms].c, out[nterms].nca);
			sign *= fremove(i_n, out[nterms].a, out[nterms].nca);
			out[nterms].nca--;
			out[nterms].p *= hop*sign;
			term_shift(out+nterms, 0);
			nterms++;
		} else if (in_array(n, t->d, t->nd) == -1 && in_array(n, t->c, t->nca) == -1) {
			out[nterms] = *t;
			out[nterms].p *= hop*freplace(i, n, out[nterms].c, out[nterms].nca);
			term_shift(out+nterms, 0);
			nterms++;
		}
	}
	for (int i = 0; i < t->nca; i++) { // a
		const uint16_t n = t->a[i] - b;
		int i_n;
		if ((i_n = in_array(n, t->c, t->nca)) >= 0) {
			out[nterms] = *t;
			finsert(n, out[nterms].d, out[nterms].nd);
			out[nterms].nd++;
			int sign = ONE_IF_EVEN(out[nterms].nca);
			sign *= fremove(i_n, out[nterms].c, out[nterms].nca);
			sign *= fremove(i, out[nterms].a, out[nterms].nca);
			out[nterms].nca--;
			out[nterms].p *= hop*sign;
			term_shift(out+nterms, 0);
			nterms++;
		} else if (in_array(n, t->d, t->nd) == -1 && in_array(n, t->a, t->nca) == -1) {
			out[nterms] = *t;
			out[nterms].p *= -hop*freplace(i, n, out[nterms].a, out[nterms].nca);
			term_shift(out+nterms, 0);
			nterms++;
		}
	}
	for (int i = 0; i < t->nd; i++) { // d
		const uint16_t o = t->d[i];
		const uint16_t n = o - b;
		int i_n;
		if ((i_n = in_array(n, t->c, t->nca)) >= 0) {
			out[nterms] = *t;
			freplace(i, n, out[nterms].d, out[nterms].nd);
			out[nterms].p *= hop*freplace(i_n, o, out[nterms].c, out[nterms].nca);
			term_shift(out+nterms, 0);
			nterms++;
		} else if (in_array(n, t->d, t->nd) == -1 && in_array(n, t->a, t->nca) == -1) {
			out[nterms] = *t;
			fremove(i, out[nterms].d, out[nterms].nd);
			out[nterms].nd--;
			int sign = ONE_IF_ODD(out[nterms].nca);
			sign *= finsert(o, out[nterms].c, out[nterms].nca);
			sign *= finsert(n, out[nterms].a, out[nterms].nca);
			out[nterms].nca++;
			out[nterms].p *= hop*sign;
			term_shift(out+nterms, 0);
			nterms++;
		}
	}
	return nterms;
}

static int term_Vcomm(const struct term *t, struct term *out, const double U)
{
	int nterms = 0;
	int selfs = 0;
	for (int i = 0; i < t->nca; i++) { // c operators
		const uint16_t n = t->c[i] ^ 0x8000;
		int i_n;
		if (in_array(n, t->d, t->nd) >= 0) { // n in d
			selfs++;
		} else if ((i_n = in_array(n, t->c, t->nca)) >= 0) { // n in c
			if (i_n > i) selfs++;
		} else if (in_array(n, t->a, t->nca) == -1) { // n not in a
			out[nterms] = *t;
			finsert(n, out[nterms].d, out[nterms].nd);
			out[nterms].nd++;
			out[nterms].p *= U;
			term_shift(out+nterms, 0);
			nterms++;
		}
	}
	for (int i = 0; i < t->nca; i++) { // a operators
		const uint16_t n = t->a[i] ^ 0x8000;
		int i_n;
		if (in_array(n, t->d, t->nd) >= 0) { // n in d
			selfs--;
		} else if ((i_n = in_array(n, t->a, t->nca)) >= 0) { // n in a
			if (i_n < i) selfs--;
		} else if (in_array(n, t->c, t->nca) == -1) { // n not in c
			out[nterms] = *t;
			finsert(n, out[nterms].d, out[nterms].nd);
			out[nterms].nd++;
			out[nterms].p *= -U;
			term_shift(out+nterms, 0);
			nterms++;
		}
	}
	if (selfs != 0) {
		out[nterms] = *t;
		out[nterms].p *= U*selfs;
		term_shift(out+nterms, 0); // this shouldn't do anything
		nterms++;
	}
	return nterms;
}

bool operator==(const struct term& t, const struct term& s)
{
	if (t.nca != s.nca) return false;
	if (t.nd != s.nd) return false;
	for (int i = 0; i < t.nca; i++)
		if (t.a[i] != s.a[i]) return false;
	for (int i = 0; i < t.nca; i++)
		if (t.c[i] != s.c[i]) return false;
	for (int i = 0; i < t.nd; i++)
		if (t.d[i] != s.d[i]) return false;
	return true;
}

struct term_hash {
	size_t operator()(const struct term& t) const // fnv-1a
	{
		uint64_t h = 14695981039346656037u;
		h ^= t.nca;
		h *= 1099511628211;
		h ^= t.nd;
		h *= 1099511628211;
		for (int i = 0; i < t.nca; i++) {
			h ^= t.a[i] & 0xff;
			h *= 1099511628211;
			h ^= t.a[i] >> 8;
			h *= 1099511628211;
		}
		for (int i = 0; i < t.nca; i++) {
			h ^= t.c[i] & 0xff;
			h *= 1099511628211;
			h ^= t.c[i] >> 8;
			h *= 1099511628211;
		}
		for (int i = 0; i < t.nd; i++) {
			h ^= t.d[i] & 0xff;
			h *= 1099511628211;
			h ^= t.d[i] >> 8;
			h *= 1099511628211;
		}
		return (size_t)h;
	}
};

// H commutator of nt terms in ts
static int terms_Hcomm(const struct term *ts, struct term *out, const int nt, const double U)
{
	std::unordered_map<struct term, double, term_hash> term_map;
#ifndef tp
	term_map.rehash(7*nt);
#else
	term_map.rehash(10*nt);
#endif
	for (int i = 0; i < nt; i++) {
		int n = 0;
#ifndef tp
		n += term_Kcomm(ts + i, out + n, -256, -1.0);
		n += term_Kcomm(ts + i, out + n, -1, -1.0);
		n += term_Kcomm(ts + i, out + n, 1, -1.0);
		n += term_Kcomm(ts + i, out + n, 256, -1.0);
#else
		n += term_Kcomm(ts + i, out + n, -257, 0.25);
		n += term_Kcomm(ts + i, out + n, -256, -1.0);
		n += term_Kcomm(ts + i, out + n, -255, 0.25);
		n += term_Kcomm(ts + i, out + n, -1, -1.0);
		n += term_Kcomm(ts + i, out + n, 1, -1.0);
		n += term_Kcomm(ts + i, out + n, 255, 0.25);
		n += term_Kcomm(ts + i, out + n, 256, -1.0);
		n += term_Kcomm(ts + i, out + n, 257, 0.25);
#endif
		n += term_Vcomm(ts + i, out + n, U);
		for (int j = 0; j < n; j++) {
			const double val = out[j].p;
			out[j].p = 0.0;
			auto x = term_map.find(out[j]);
			if (x != term_map.end()) {
				x->second += val;
			} else {
				term_map.insert(std::make_pair(out[j], val));
			}
		}
	}

	int nout = 0;
	for (const auto& x : term_map) {
		if (x.second != 0.0) {
			out[nout] = x.first;
			out[nout++].p = x.second;
		}
	}

	std::sort(out, out+nout, [](const struct term t, const struct term s) {
		if (t.nca != s.nca) return t.nca < s.nca;
		for (int i = 0; i < t.nca; i++)
			if (t.a[i] != s.a[i]) return t.a[i] < s.a[i];
		for (int i = 0; i < t.nca; i++)
			if (t.c[i] != s.c[i]) return t.c[i] < s.c[i];
		if (t.nd != s.nd) return t.nd < s.nd;
		for (int i = 0; i < t.nd; i++)
			if (t.d[i] != s.d[i]) return t.d[i] < s.d[i];
		return false;
	});
	return nout;
}

// assumes sorted a, b
static inline int n_union(const uint16_t *a, const uint16_t *b, const int na, const int nb)
{
	int inter = 0;
	for (int ia = 0, ib = 0; ia < na && ib < nb;) {
		if (a[ia] < b[ib]) {
			ia++;
		} else if (a[ia] > b[ib]) {
			ib++;
		} else {
			ia++;
			ib++;
			inter++;
		}
	}
	return na + nb - inter;
}

// assumes t->nca == s->nca == 0
// too complicated to comment
static kdouble ev_weird(const struct term *t, const struct term *s, const double *pow_n)
{
	int16_t diffs[MAX_OP*MAX_OP];
	int ndiffs = 0;
	for (int i = 0; i < t->nd; i++)
		for (int j = 0; j < s->nd; j++)
			if (TO_S(t->d[i]) == TO_S(s->d[j]))
				diffs[ndiffs++] = (t->d[i] & 0x7fff) - (s->d[j] & 0x7fff);
	std::sort(diffs, diffs + ndiffs);
	kdouble total = 0.0;
	int prev = 0;
	for (int d = 1; d < ndiffs; d++) {
		if (diffs[d] != diffs[d - 1]) {
			total += 1.0/pow_n[d - prev];
			total -= 1.0;
			prev = d;
		}
	}
	total += 1.0/pow_n[ndiffs - prev];
	total -= 1.0;
	return total * (pow_n[t->nd + s->nd] * t->p * s->p);
}

// assumes t->nca > 0, t->c == s->a, and t->a == s->c
// multiply by n(1-n)^nca outside here
static inline double ev_normal(const struct term *t, const struct term *s, const double *pow_n)
{
	return pow_n[n_union(t->d, s->d, t->nd, s->nd)] * t->p * s->p;
}

static int term_ev_cmp(const struct term *const __restrict t, const struct term *const __restrict s)
{
	if (t->nca != s->nca) return t->nca - s->nca;
	for (int i = 0; i < t->nca; i++)
		if (t->a[i] != s->c[i]) return t->a[i] - s->c[i];
	for (int i = 0; i < t->nca; i++)
		if (t->c[i] != s->a[i]) return t->c[i] - s->a[i];
	return 0;
}
// assumes t and s have been shifted oppositely
static kdouble ev_terms(const struct term *t, const struct term *s, const int nt, const int ns,
	const double *pow_n, const double *pow_n1n)
{
	kdouble total = 0.0;
	int it = 0, is = 0;

	// weird terms
	if (t[it].nca == 0 && s[is].nca == 0) {
		int it_next, is_next;
		for (it_next = it; it_next < nt; it_next++)
			if (t[it_next].nca != 0) break;
		for (is_next = is; is_next < ns; is_next++)
			if (s[is_next].nca != 0) break;
		for (int jt = it; jt < it_next; jt++)
			for (int js = is; js < is_next; js++)
				total += ev_weird(t + jt, s + js, pow_n);
		it = it_next;
		is = is_next;
	}

	// normal terms
	while (it < nt && is < ns) {
		int cmp = term_ev_cmp(t + it, s + is);
		if (cmp < 0) {
			it++;
		} else if (cmp > 0) {
			is++;
		} else {
			kdouble temp = 0.0;
			int it_next, is_next;
			for (it_next = it; it_next < nt; it_next++)
				if (term_ev_cmp(t + it_next, s + is) != 0) break;
			for (is_next = is; is_next < ns; is_next++)
				if (term_ev_cmp(t + it, s + is_next) != 0) break;
			for (int jt = it; jt < it_next; jt++)
				for (int js = is; js < is_next; js++)
					temp += ev_normal(t + jt, s + js, pow_n);
			total += temp * pow_n1n[t[it].nca];
			it = it_next;
			is = is_next;
		}
	}
	return total;
}

static kdouble ev_jj(const struct term *jt, const struct term *js, struct term *buf, const int nt, const int ns,
	const double *pow_n, const double *pow_n1n)
{
	kdouble total = 0.0;
	for (int i = 0; i < ns; i++) buf[i] = js[i];

	// jpu jpu
	for (int i = 0; i < ns; i++)
		term_shift(buf + i, 1);
	std::sort(buf, buf+ns, [](const struct term t, const struct term s) {
		if (t.nca != s.nca) return t.nca < s.nca;
		for (int i = 0; i < t.nca; i++)
			if (t.c[i] != s.c[i]) return t.c[i] < s.c[i];
		for (int i = 0; i < t.nca; i++)
			if (t.a[i] != s.a[i]) return t.a[i] < s.a[i];
		return false;
	});
	total += ev_terms(jt, buf, nt, ns, pow_n, pow_n1n);

	// jpu jpd
	for (int i = 0; i < ns; i++) {
		term_flip(buf + i);
		term_shift(buf + i, 1);
	}
	std::sort(buf, buf+ns, [](const struct term t, const struct term s) {
		if (t.nca != s.nca) return t.nca < s.nca;
		for (int i = 0; i < t.nca; i++)
			if (t.c[i] != s.c[i]) return t.c[i] < s.c[i];
		for (int i = 0; i < t.nca; i++)
			if (t.a[i] != s.a[i]) return t.a[i] < s.a[i];
		return false;
	});
	total += ev_terms(jt, buf, nt, ns, pow_n, pow_n1n);

	// jpu jnu
	for (int i = 0; i < ns; i++) {
		term_flip_invert(buf + i);
		term_shift(buf + i, 1);
	}
	std::sort(buf, buf+ns, [](const struct term t, const struct term s) {
		if (t.nca != s.nca) return t.nca < s.nca;
		for (int i = 0; i < t.nca; i++)
			if (t.c[i] != s.c[i]) return t.c[i] < s.c[i];
		for (int i = 0; i < t.nca; i++)
			if (t.a[i] != s.a[i]) return t.a[i] < s.a[i];
		return false;
	});
	total -= ev_terms(jt, buf, nt, ns, pow_n, pow_n1n);

	// jpu jnd
	for (int i = 0; i < ns; i++) {
		term_flip(buf + i);
		term_shift(buf + i, 1);
	}
	std::sort(buf, buf+ns, [](const struct term t, const struct term s) {
		if (t.nca != s.nca) return t.nca < s.nca;
		for (int i = 0; i < t.nca; i++)
			if (t.c[i] != s.c[i]) return t.c[i] < s.c[i];
		for (int i = 0; i < t.nca; i++)
			if (t.a[i] != s.a[i]) return t.a[i] < s.a[i];
		return false;
	});
	total -= ev_terms(jt, buf, nt, ns, pow_n, pow_n1n);

	return total;
}

int main(int argc, char **argv)
{
	if (argc < 4) return 0;
	double UU = atof(argv[1]);
	double nn = atof(argv[2]);
	int order = atoi(argv[3]);


        long long int num0 = 100;
        long long int num1 = 2000;
        //long long int num1 = 1500;
        long long int num2 = 1000000;
        long long int length0 = num0*num2;
        long long int length = num1*num2;
        struct term *buf = (struct term *)aligned_alloc(64, length0 * sizeof(struct term));
        struct term *bufe = (struct term *)aligned_alloc(64, length * sizeof(struct term));
	struct term *bufk = (struct term *)aligned_alloc(64, length * sizeof(struct term));
        struct term *bufp = (struct term *)aligned_alloc(64, length * sizeof(struct term));
        //struct term *buf = (struct term *)aligned_alloc(64, 10000 * 1000000 * sizeof(struct term));
        //struct term *bufe = (struct term *)aligned_alloc(64, 10000 * 1000000 * sizeof(struct term));
	
        struct term *jpu[16] = {0};
        struct term *jpue[16] = {0};
        struct term *jpuk[16] = {0};	
        struct term *jpup[16] = {0};
        int len[16] = {0};
        int lene[16] = {0};
        int lenk[16] = {0};
        int lenp[16] = {0};



#ifndef tp
	buf[0] = (struct term) {
		.p = 1.0,
		.d = {0}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
		.nd = 0, .nca = 1
	};
	jpu[0] = buf;
	len[0] = 1;
#else
	buf[0] = (struct term) {
		.p = tp,
		.d = {0}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
		.nd = 0, .nca = 1
	};
	buf[1] = (struct term) {
		.p = 1.0,
		.d = {0}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
		.nd = 0, .nca = 1
	};
	buf[2] = (struct term) {
		.p = tp,
		.d = {0}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
		.nd = 0, .nca = 1
	};
	jpu[0] = buf;
	len[0] = 3;
#endif

#ifndef tp
        bufe[0] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 1, -1 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[4] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 1, 1 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[5] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 2, 0 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[2] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 0, 0 )}, .c = {TO_SXY(0, 1, 0 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufe[3] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 1, 0 )}, .c = {TO_SXY(0, 1, 0 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufe[1] = (struct term) {
                .p = -UU/2,
                .d = {0}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        jpue[0] = bufe;
        lene[0] = 6;
#else
        bufe[4] = (struct term) {
                .p = -2.0*tp-UU/2,
                .d = {0}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[7] = (struct term) {
                .p = -1.0-UU/2*tp,
                .d = {0}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[1] = (struct term) {
                .p = -1.0-UU/2*tp,
                .d = {0}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[13] = (struct term) {
                .p = -(1.0+2*tp*tp),
                .d = {0}, .c = {TO_SXY(0, 2, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[14] = (struct term) {
                .p = -2.0*tp,
                .d = {0}, .c = {TO_SXY(0, 2, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[12] = (struct term) {
                .p = -2.0*tp,
                .d = {0}, .c = {TO_SXY(0, 2, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[10] = (struct term) {
                .p = -tp,
                .d = {0}, .c = {TO_SXY(0, 1, 2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[0] = (struct term) {
                .p = -tp,
                .d = {0}, .c = {TO_SXY(0, 1, -2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[15] = (struct term) {
                .p = -tp*tp,
                .d = {0}, .c = {TO_SXY(0, 2, 2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[11] = (struct term) {
                .p = -tp*tp,
                .d = {0}, .c = {TO_SXY(0, 2, -2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufe[2] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 0, 0)}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufe[3] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 1, -1)}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufe[8] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 0, 0)}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufe[9] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 1, 1)}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufe[5] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 0, 0)}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufe[6] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 1, 0)}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        jpue[0] = bufe;
        lene[0] = 16;
#endif


#ifndef tp
        bufk[0] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 1, -1 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[1] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 1, 1 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[2] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 2, 0 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        jpuk[0] = bufk;
        lenk[0] = 3;
#else
        bufk[2] = (struct term) {
                .p = -2.0*tp,
                .d = {0}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[3] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[1] = (struct term) {
                .p = -1.0,
                .d = {0}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[7] = (struct term) {
                .p = -(1.0+2*tp*tp),
                .d = {0}, .c = {TO_SXY(0, 2, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[8] = (struct term) {
                .p = -2.0*tp,
                .d = {0}, .c = {TO_SXY(0, 2, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[6] = (struct term) {
                .p = -2.0*tp,
                .d = {0}, .c = {TO_SXY(0, 2, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[4] = (struct term) {
                .p = -tp,
                .d = {0}, .c = {TO_SXY(0, 1, 2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[0] = (struct term) {
                .p = -tp,
                .d = {0}, .c = {TO_SXY(0, 1, -2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[9] = (struct term) {
                .p = -tp*tp,
                .d = {0}, .c = {TO_SXY(0, 2, 2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufk[5] = (struct term) {
                .p = -tp*tp,
                .d = {0}, .c = {TO_SXY(0, 2, -2)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        jpuk[0] = bufk;
        lenk[0] = 10;
#endif
        

#ifndef tp
        bufp[1] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 0, 0 )}, .c = {TO_SXY(0, 1, 0 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[2] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 1, 0 )}, .c = {TO_SXY(0, 1, 0 )}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[0] = (struct term) {
                .p = -UU/2,
                .d = {0}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };

        jpup[0] = bufp;
        lenp[0] = 3;
#else
        bufp[1] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 0, 0)}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[2] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 1, -1)}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[7] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 0, 0)}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[8] = (struct term) {
                .p = UU/2*tp,
                .d = {TO_SXY(1, 1, 1)}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[4] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 0, 0)}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[5] = (struct term) {
                .p = UU/2,
                .d = {TO_SXY(1, 1, 0)}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 1, .nca = 1
        };
        bufp[3] = (struct term) {
                .p = -UU/2,
                .d = {0}, .c = {TO_SXY(0, 1, 0)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufp[6] = (struct term) {
                .p = -UU/2*tp,
                .d = {0}, .c = {TO_SXY(0, 1, 1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };
        bufp[0] = (struct term) {
                .p = -UU/2*tp,
                .d = {0}, .c = {TO_SXY(0, 1, -1)}, .a = {TO_SXY(0, 0, 0)},
                .nd = 0, .nca = 1
        };

        jpup[0] = bufp;
        lenp[0] = 9;
#endif



        printf("U=%f, n=%f\n", UU, nn);
        printf("for j\n");
	printf("len[0]=%d\n", len[0]);
	for (int i = 1; i <= order; i++) {
		jpu[i] = jpu[i-1] + len[i-1];
		len[i] = terms_Hcomm(jpu[i-1], jpu[i], len[i-1], UU);
		printf("len[%d]=%d\n", i, len[i]);
	}

	for (int i = 0; i <= order; i++) {
		int max_nd = 0, max_nca = 0;
		double max_p = 0.0;
		for (int it = 0; it < len[i]; it++) {
			if (jpu[i][it].nd > max_nd) max_nd = jpu[i][it].nd;
			if (jpu[i][it].nca > max_nca) max_nca = jpu[i][it].nca;
			if (fabs(jpu[i][it].p) > max_p) max_p = fabs(jpu[i][it].p);
		}
		printf("%d: %d, %d, %f\n", i, max_nd, max_nca, max_p);
	}
        
        printf("for je\n");
        printf("lene[0]=%d\n", lene[0]);
        for (int i = 1; i <= order; i++) {
                jpue[i] = jpue[i-1] + lene[i-1];
                lene[i] = terms_Hcomm(jpue[i-1], jpue[i], lene[i-1], UU);
                printf("lene[%d]=%d\n", i, lene[i]);
        }

        for (int i = 0; i <= order; i++) {
                int max_nd = 0, max_nca = 0;
                double max_p = 0.0;
                for (int it = 0; it < lene[i]; it++) {
                        if (jpue[i][it].nd > max_nd) max_nd = jpue[i][it].nd;
                        if (jpue[i][it].nca > max_nca) max_nca = jpue[i][it].nca;
                        if (fabs(jpue[i][it].p) > max_p) max_p = fabs(jpue[i][it].p);
                }
                printf("%d: %d, %d, %f\n", i, max_nd, max_nca, max_p);
        }

        printf("for jk\n");
        printf("lenk[0]=%d\n", lenk[0]);
        for (int i = 1; i <= order; i++) {
                jpuk[i] = jpuk[i-1] + lenk[i-1];
                lenk[i] = terms_Hcomm(jpuk[i-1], jpuk[i], lenk[i-1], UU);
                printf("lenk[%d]=%d\n", i, lenk[i]);
        }

        for (int i = 0; i <= order; i++) {
                int max_nd = 0, max_nca = 0;
                double max_p = 0.0;
                for (int it = 0; it < lenk[i]; it++) {
                        if (jpuk[i][it].nd > max_nd) max_nd = jpuk[i][it].nd;
                        if (jpuk[i][it].nca > max_nca) max_nca = jpuk[i][it].nca;
                        if (fabs(jpuk[i][it].p) > max_p) max_p = fabs(jpuk[i][it].p);
                }
                printf("%d: %d, %d, %f\n", i, max_nd, max_nca, max_p);
        }

        printf("for jp\n");
        printf("lenp[0]=%d\n", lenp[0]);
        for (int i = 1; i <= order; i++) {
                jpup[i] = jpup[i-1] + lenp[i-1];
                lenp[i] = terms_Hcomm(jpup[i-1], jpup[i], lenp[i-1], UU);
                printf("lenp[%d]=%d\n", i, lenp[i]);
        }
        
        for (int i = 0; i <= order; i++) {
                int max_nd = 0, max_nca = 0;
                double max_p = 0.0;
                for (int it = 0; it < lenp[i]; it++) {
                        if (jpup[i][it].nd > max_nd) max_nd = jpup[i][it].nd;
                        if (jpup[i][it].nca > max_nca) max_nca = jpup[i][it].nca;
                        if (fabs(jpup[i][it].p) > max_p) max_p = fabs(jpup[i][it].p);
                }
                printf("%d: %d, %d, %f\n", i, max_nd, max_nca, max_p);
        }

	// powers of n
	double pow_n[2*MAX_OP + 1], pow_n1n[MAX_OP + 1];
	pow_n[0] = 1.0;
	for (int i = 1; i < 2*MAX_OP + 1; i++) pow_n[i] = nn*pow_n[i-1];
	pow_n1n[0] = 1.0;
	for (int i = 1; i < MAX_OP + 1; i++) pow_n1n[i] = nn*(1.0-nn)*pow_n1n[i-1];

	// expectation values
	printf("<jj>");
	struct term *buf2 = (struct term *)aligned_alloc(64, (len[order] + 256) * sizeof(struct term));
	if (argc == 4) {
		for (int i = 0; i <= order; i++) {
			kdouble jj = ev_jj(jpu[i], jpu[i], buf2, len[i], len[i], pow_n, pow_n1n);
			printf("%.20f,\n", jj.val);
			// printf("%.20f + %.20f,\n", jj.val, jj.err);
		}
	} else if (argc == 6) {
		int i = atoi(argv[4]);
		int j = atoi(argv[5]);
		kdouble jj = ev_jj(jpu[i], jpu[j], buf2, len[i], len[j], pow_n, pow_n1n);
		printf("<j[%d]j[%d]>= %.20f + %.20f\n", i, j, jj.val, jj.err);
	}

	free(buf2);


        printf("<jeje>");
        struct term *buf4 = (struct term *)aligned_alloc(64, (lene[order] + 256) * sizeof(struct term));
        if (argc == 4) {
                for (int i = 0; i <= order; i++) {
                        kdouble jeje = ev_jj(jpue[i], jpue[i], buf4, lene[i], lene[i], pow_n, pow_n1n);
                        printf("%.20f,\n", jeje.val);
                        // printf("%.20f + %.20f,\n", jj.val, jj.err);
                }
        } else if (argc == 6) {
                int i = atoi(argv[4]);
                int j = atoi(argv[5]);
                kdouble jeje = ev_jj(jpue[i], jpue[j], buf4, lene[i], lene[j], pow_n, pow_n1n);
                printf("<je[%d]je[%d]>= %.20f + %.20f\n", i, j, jeje.val, jeje.err);
        }

        free(buf4);
        


        printf("<jkjk>");
        struct term *buf3 = (struct term *)aligned_alloc(64, (lenk[order] + 256) * sizeof(struct term));
        if (argc == 4) {
                for (int i = 0; i <= order; i++) {
                        kdouble jkjk = ev_jj(jpuk[i], jpuk[i], buf3, lenk[i], lenk[i], pow_n, pow_n1n);
                        printf("%.20f,\n", jkjk.val);
                        // printf("%.20f + %.20f,\n", jj.val, jj.err);
                }
        } else if (argc == 6) {
                int i = atoi(argv[4]);
                int j = atoi(argv[5]);
                kdouble jkjk = ev_jj(jpuk[i], jpuk[j], buf3, lenk[i], lenk[j], pow_n, pow_n1n);
                printf("<jk[%d]jk[%d]>= %.20f + %.20f\n", i, j, jkjk.val, jkjk.err);
        }
	free(buf3);


        printf("<jpjp>");
        struct term *buf5 = (struct term *)aligned_alloc(64, (lenp[order] + 256) * sizeof(struct term));
        if (argc == 4) {
                for (int i = 0; i <= order; i++) {
                        kdouble jpjp = ev_jj(jpup[i], jpup[i], buf5, lenp[i], lenp[i], pow_n, pow_n1n);
                        printf("%.20f,\n", jpjp.val);
                        // printf("%.20f + %.20f,\n", jj.val, jj.err);
                }
        } else if (argc == 6) {
                int i = atoi(argv[4]);
                int j = atoi(argv[5]);
                kdouble jpjp = ev_jj(jpup[i], jpup[j], buf5, lenp[i], lenp[j], pow_n, pow_n1n);
                printf("<jp[%d]jp[%d]>= %.20f + %.20f\n", i, j, jpjp.val, jpjp.err);
        }

        free(buf5);

        free(buf);
	return 0;
}

	// uint16_t a[] = {1,3,5,7,9,17};
	// uint16_t b[] = {2,4,9,11};
	// printf("%d\n", n_union(a, b, sizeof(a)/sizeof(a[0]), sizeof(b)/sizeof(b[0])));

	//for (int i = 0; i < len[3]; i++) term_print(jpu[3] + i);

	// printf("%lu\n", sizeof(struct term));
	// uint16_t a = TO_SXY(0,-3,-14);
	// printf("%d %d %d\n", TO_S(a), TO_X(a), TO_Y(a));

	// struct term t = {0};
	// t.d[0] = TO_SXY(1, -9, 13);
	// t.d[1] = TO_SXY(1, -3, 11);
	// t.c[0] = TO_SXY(0, 4, -3);
	// t.c[1] = TO_SXY(0, 7, 0);
	// t.c[2] = TO_SXY(1, 3, 0);
	// t.a[0] = TO_SXY(0, -4, 1);
	// t.a[1] = TO_SXY(0, -2, -1);
	// t.a[2] = TO_SXY(1, -2, -1);
	// t.nd = 2; t.nca = 3;
	// term_print(&t);
	// term_flip_invert(&t);
	// term_print(&t);
	// term_shift(&t, 0);
	// term_print(&t);
	// term_shift(&t, 1);
	// term_print(&t);
	// struct term s = t;
	// term_shift(&s, 0);
	// term_print(&s);
	// term_print(&t);
	// term_shift(&t, 0);
	// printf("%d\n", term_eq(&t, &s));

	// uint16_t a[] = {1,3,5,7,9,11,0,0};
	// //printf("%d\n", in_array(125, a, 6));
	// //for (int i = 0; i < 8; i++) printf("%d,",a[i]); printf("\n");
	// //finsert(4211,a,6);
	// for (int i = 0; i < 8; i++) printf("%d,",a[i]); printf("\n");
	// //printf("%d\n", freplace(0, 3000, a, 6));
	// //printf("%d\n", finsert(10, a, 6));
	// printf("%d\n", fremove(11, a, 6));
	// for (int i = 0; i < 8; i++) printf("%d,",a[i]); printf("\n");

	// struct term jpu0 = {0};
	// jpu0.p = 1;
	// jpu0.c[0] = TO_SXY(0, 1, 0);
	// jpu0.a[0] = TO_SXY(0, 0, 0);
	// jpu0.nca = 1;
