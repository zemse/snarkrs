// Does a short-tagged value behave like the long-tagged form of the same value?
//
// `patch_fr_generic` in build.rs lets mul_s1s2 keep a non-negative product short where
// the shipped code widened it to Fr_LONG. That is only sound if every consumer of the
// result agrees on both representations, because the choice propagates: a short product
// feeds adds, compares, shifts and divisions, and eventually a witness file.
//
// So build both forms of the same value and compare what each consumer returns. Compare
// by value, not by bytes: shortVal is dead storage when the type is Fr_LONG and the two
// paths legitimately leave different junk there. What must agree is the type tag and
// whichever field that tag says is live, which is all any consumer and storeBinWitness
// ever read.
//
// Negative values are excluded deliberately, and that exclusion is the point of the
// test. Build it with -DG16_ALLOW_NEGATIVE to include them and it fails, which is the
// evidence for the restriction in the patch: Fr_sub of a negative short and a long comes
// back unreduced where the all-long path reduces.
#include "fr.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

extern "C" void Fr_rawCopyS2L(FrRawElement pRawResult, int64_t val);

static FrElement as_short(int64_t v) {
    FrElement e; memset(&e, 0, sizeof e);
    e.type = Fr_SHORT; e.shortVal = (int32_t)v; return e;
}
static FrElement as_long(int64_t v) {
    FrElement e; memset(&e, 0, sizeof e);
    e.type = Fr_LONG; Fr_rawCopyS2L(e.longVal, v); return e;
}

static long checks = 0;
static int fails = 0;

static bool same(const FrElement* a, const FrElement* b) {
    if (a->type != b->type) return false;
    if (a->type & Fr_LONG) return memcmp(a->longVal, b->longVal, sizeof(a->longVal)) == 0;
    return a->shortVal == b->shortVal;
}
static void cmp(const char* what, int64_t v, int64_t w, FrElement* a, FrElement* b) {
    checks++;
    if (!same(a, b) && fails++ < 8)
        printf("  %s(%lld, %lld): short path %08x/%016llx, long path %08x/%016llx\n",
               what, (long long)v, (long long)w,
               a->type, (unsigned long long)a->longVal[0],
               b->type, (unsigned long long)b->longVal[0]);
}
static void cmpi(const char* what, int64_t v, int64_t w, int64_t a, int64_t b) {
    checks++;
    if (a != b && fails++ < 8)
        printf("  %s(%lld, %lld): short path %lld, long path %lld\n",
               what, (long long)v, (long long)w, (long long)a, (long long)b);
}
// Every comparison is made on the normal form, which is what reaches the witness.
#define CMP(op, fn)                                                         \
    do { FrElement rs, rl, ns, nl;                                          \
         fn(&rs, &s, o); fn(&rl, &l, o);                                    \
         Fr_toLongNormal(&ns, &rs); Fr_toLongNormal(&nl, &rl);              \
         cmp(op, v, w, &ns, &nl); } while (0)
#define CMPB(op, fn)                                                        \
    do { FrElement rs, rl; fn(&rs, &s, o); fn(&rl, &l, o);                  \
         cmpi(op, v, w, Fr_isTrue(&rs), Fr_isTrue(&rl)); } while (0)

int main() {
    // The shapes these circuits actually produce, plus the edges of the short range.
    static const int64_t POOL[] = {
        0, 1, 2, 3, 5, 6, 7, 24, 64, 255, 256, 65535, 1000000,
        1073741824LL, 2147483647LL,
#ifdef G16_ALLOW_NEGATIVE
        -1, -2, -6, -64, -1073741823LL, -2147483648LL,
#endif
    };
    const int N = (int)(sizeof(POOL) / sizeof(POOL[0]));

    for (int i = 0; i < N; i++) {
        const int64_t v = POOL[i];
        FrElement s = as_short(v), l = as_long(v);

        FrElement ns, nl;
        Fr_toLongNormal(&ns, &s); Fr_toLongNormal(&nl, &l);
        cmp("toLongNormal", v, 0, &ns, &nl);
        cmpi("toInt", v, 0, Fr_toInt(&s), Fr_toInt(&l));

        for (int j = 0; j < N; j++) {
            const int64_t w = POOL[j];
            FrElement o_s = as_short(w), o_l = as_long(w);
            for (int ot = 0; ot < 2; ot++) {
                FrElement* o = ot ? &o_l : &o_s;
                CMP("add",  Fr_add);
                CMP("sub",  Fr_sub);
                CMP("mul",  Fr_mul);
                CMP("band", Fr_band);
                CMP("bor",  Fr_bor);
                CMP("bxor", Fr_bxor);
                CMPB("eq",  Fr_eq);
                CMPB("neq", Fr_neq);
                CMPB("lt",  Fr_lt);
                CMPB("gt",  Fr_gt);
                CMPB("leq", Fr_leq);
                CMPB("geq", Fr_geq);
                if (w > 0) { CMP("mod", Fr_mod); CMP("idiv", Fr_idiv); }
                if (w >= 0 && w < 254) { CMP("shr", Fr_shr); }
                if (w >= 0 && w < 32)  { CMP("pow", Fr_pow); }
            }
        }
    }
    printf("%s: %ld checks, %d mismatches\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
