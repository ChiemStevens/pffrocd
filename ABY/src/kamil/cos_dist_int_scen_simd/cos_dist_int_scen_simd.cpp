/**
 \file 		abyfloat.cpp
 \author	daniel.demmler@ec-spride.de
 \copyright	ABY - A Framework for Efficient Mixed-protocol Secure Two-party Computation
 Copyright (C) 2019 Engineering Cryptographic Protocols Group, TU Darmstadt
			This program is free software: you can redistribute it and/or modify
			it under the terms of the GNU Lesser General Public License as published
			by the Free Software Foundation, either version 3 of the License, or
			(at your option) any later version.
			ABY is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
			MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
			GNU Lesser General Public License for more details.
			You should have received a copy of the GNU Lesser General Public License
			along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include "../../abycore/aby/abyparty.h"
#include "../../abycore/circuit/share.h"
#include "../../abycore/circuit/booleancircuits.h"
#include "../../abycore/circuit/arithmeticcircuits.h"
#include "../../abycore/sharing/sharing.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <fstream>




void read_test_options(int32_t *argcp, char ***argvp, e_role *role,
					   uint32_t *bitlen, uint32_t *nvals, uint32_t *secparam, std::string *address,
					   uint16_t *port, int32_t *test_op, uint32_t *test_bit, e_mt_gen_alg *mt_alg, uint32_t *debug, std::string *inputfile, std::string *pffrocd_path)
{

	uint32_t int_role = 0, int_port = 0, int_testbit = 0, int_mt_alg = 0;

	parsing_ctx options[] =
		{{(void *)&int_role, T_NUM, "r", "Role: 0/1", true, false},
		 {(void *)&int_testbit, T_NUM, "i", "test bit", false, false},
		 {(void *)nvals, T_NUM, "n", "Number of parallel operation elements", false, false},
		 {(void *)bitlen, T_NUM, "b", "Bit-length, default 32", false, false},
		 {(void *)secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false},
		 {(void *)address, T_STR, "a", "IP-address, default: localhost", false, false},
		 {(void *)&int_port, T_NUM, "p", "Port, default: 7766", false, false},
		 {(void *)test_op, T_NUM, "t", "Single test (leave out for all operations), default: off", false, false},
		 {(void *)&int_mt_alg, T_NUM, "x", "Arithmetic multiplication triples algorithm", false, false},
		 {(void *)debug, T_NUM, "d", "debug mode (more printing) (0/1)", false, false},
		 {(void *)inputfile, T_STR, "f", "Input file containing face embeddings", true, false},
		 {(void *)pffrocd_path, T_STR, "o", "absolute path to pffrocd directory", true, false}
		};
	std::cout << "Start pre checks" << std::endl; 
	if (!parse_options(argcp, argvp, options,
					   sizeof(options) / sizeof(parsing_ctx)))
	{
		print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
		std::cout << "Exiting" << std::endl;
		exit(0);
	}

	assert(int_role < 2);
	*role = (e_role)int_role;

	if (int_port != 0)
	{
		assert(int_port < 1 << (sizeof(uint16_t) * 8));
		*port = (uint16_t)int_port;
	}

	*test_bit = int_testbit;

	if (int_mt_alg == 0) {
		*mt_alg = MT_OT;
	} else if (int_mt_alg == 1) {
		*mt_alg = MT_PAILLIER;
	} else if (int_mt_alg == 2) {
		*mt_alg = MT_DGK;
	} else {
		std::cout << "Invalid MT algorithm" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Finished pre checks" << std::endl; 
}

share* BuildInnerProductCircuit(share *s_x, share *s_y, uint32_t numbers, ArithmeticCircuit *ac) {
	uint32_t i;

	// pairwise multiplication of all input values
	s_x = ac->PutMULGate(s_x, s_y);

	// split SIMD gate to separate wires (size many)
	s_x = ac->PutSplitterGate(s_x);

	// add up the individual multiplication results and store result on wire 0
	// in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
	for (i = 1; i < numbers; i++) {
		s_x->set_wire_id(0, ac->PutADDGate(s_x->get_wire_id(0), s_x->get_wire_id(i)));
	}

	// discard all wires, except the addition result
	s_x->set_bitlength(0);

	return s_x;
}

void test_verilog_add64_SIMD(e_role role, const std::string &address, uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
							 e_mt_gen_alg mt_alg, e_sharing sharing, uint32_t debug, std::string inputfile, std::string pffrocd_path)
{
	std::cout << "Start of test" << std::endl; 
	// std::cout << "SEC LEVEL: " << seclvl.symbits << std::endl;
	// std::cout << "MT_ALG: " << mt_alg << std::endl;

	// 32 bit version operating on floats
	uint32_t bitlen = 32;

	// two arrays of real-world embeddings
	std::vector<float> xembeddings;
	std::vector<float> yembeddings;

	// array for the Sy<role> share
	std::vector<float> share_embeddings;
	std::vector<float> share_embeddings_prime;


	// reading the non-xored embeddings, i.e. current face and database face

	std::fstream infile(inputfile);

	// std::cout << "INPUT FILE NAME: " << inputfile << std::endl;

	float x, y;

	// std::cout << "starting reading x and y" << std::endl;

	while (infile >> x >> y) {
		// std::cout << "x: " << x << " | y: "<< y << std::endl;
		xembeddings.push_back(x);
		yembeddings.push_back(y);
	}

	// std::cout<<"finished reading x and y" << std::endl;

	assert(xembeddings.size() == nvals);
	assert(yembeddings.size() == nvals);

	// std::cout << "nvals" << nvals << std::endl;
	// std::cout << "xembeddings: " << xembeddings.size() << std::endl;

	// // reading the xored embedding, i.e. either Sy<0> or Sy<1> depending on the role

	// // char *fname = (char *) malloc(150); // file name buffer 
    // // sprintf(fname, "/home/dietpi/pffrocd/ABY/build/bin/share%d.txt", role);

	std::string fname = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + ".txt";
	std::string fnameprime = pffrocd_path + "/ABY/build/bin/share" + std::to_string(role) + "prime.txt";
	// //std::cout << "FNAME: " << fname << std::endl; 

	std::fstream infile_share(fname);
	std::fstream infile_share_prime(fnameprime);

	float z;
	// // std::cout << "starting reading z" << std::endl;

	while(infile_share >> z) {
		//std::cout << "z: " << z << std::endl;
		share_embeddings.push_back(z);
	}
	z = 0;
	while(infile_share_prime >> z) {
		//std::cout << "z: " << z << std::endl;
		share_embeddings_prime.push_back(z);
	}

	// //std::cout<<"finished reading z" << std::endl;
	// std::cout << "share_embeddings size: " << share_embeddings.size() << std::endl;
	// std::cout << "share embedddings 0: " << share_embeddings[0] << std::endl;
	// assert(share_embeddings.size() == nvals);
	// assert(share_embeddings_prime.size() == nvals);

	std::string circuit_dir = pffrocd_path + "/ABY/bin/circ/";

	std::cout << "CIRCUIT DIRECTORY: " << circuit_dir << std::endl;

	ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg, 100000, circuit_dir);

	std::cout << "party created" << std::endl;


	std::vector<Sharing *> &sharings = party->GetSharings();

	BooleanCircuit *bc = (BooleanCircuit *)sharings[S_BOOL]->GetCircuitBuildRoutine();
	ArithmeticCircuit *ac = (ArithmeticCircuit *)sharings[S_ARITH]->GetCircuitBuildRoutine();
	Circuit *yc = (BooleanCircuit *)sharings[S_YAO]->GetCircuitBuildRoutine();

	/**
	 Step 4: Creating the share objects - s_x_vec, s_y_vec which
	 are used as inputs to the computation. Also, s_out which stores the output.
	 */

	share *s_x_vec, *s_y_vec, *s_out;

	/**
	 Step 5: Allocate the xvals and yvals that will hold the plaintext values.
	 */
	uint16_t x1, y1;

	uint16_t output, v_sum = 0;

	std::vector<uint16_t> xvals(nvals);
	std::vector<uint16_t> yvals(nvals);

	uint32_t i;
	srand(time(NULL));

	/**
	 Step 6: Fill the arrays xvals and yvals with the generated random values.
	 Both parties use the same seed, to be able to verify the
	 result. In a real example each party would only supply
	 one input value. Copy the randomly generated vector values into the respective
	 share objects using the circuit object method PutINGate().
	 Also mention who is sharing the object.
	 The values for the party different from role is ignored,
	 but PutINGate() must always be called for both roles.
	 */
	for (i = 0; i < nvals; i++) {

		x1 = rand();
		y1 = rand();

		v_sum += x1 * y1;

		xvals[i] = x1;
		yvals[i] = y1;
	}

	s_x_vec = ac->PutSIMDINGate(nvals, xvals.data(), 64, SERVER);
	s_y_vec = ac->PutSIMDINGate(nvals, yvals.data(), 64, CLIENT);
	// s_x_vec = ac->PutSharedSIMDINGate(nvals, sharevals_prime, bitlen);
	// s_y_vec = ac->PutSharedSIMDINGate(nvals, sharevals, bitlen);
	/**
	 Step 7: Call the build method for building the circuit for the
	 problem by passing the shared objects and circuit object.
	 Don't forget to type cast the circuit object to type of share
	 */
	s_out = BuildInnerProductCircuit(s_x_vec, s_y_vec, nvals,
			(ArithmeticCircuit*) ac);

	/**
	 Step 8: Output the value of s_out (the computation result) to both parties
	 */
	s_out = ac->PutOUTGate(s_out, ALL);

	/**
	 Step 9: Executing the circuit using the ABYParty object evaluate the
	 problem.
	 */
	party->ExecCircuit();

	/**
	 Step 10: Type caste the plaintext output to 16 bit unsigned integer.
	 */
	output = s_out->get_clear_value<uint16_t>();

	std::cout << "\nCircuit Result: " << output;
	std::cout << "\nVerification Result: " << v_sum << std::endl;







	// std::cout << "circuit retrieved" << std::endl;
	// // std::cout << "circuit retrieved" << std::endl;
	// // std::cout << "here 1" << std::endl;

	// // arrays of integer pointers to doubles
	// uint32_t xvals[nvals];
	// uint32_t yvals[nvals];
	// uint32_t sharevals[nvals];
	// uint32_t sharevals_prime[nvals];
	// // std::cout << "here 1" << std::endl;

	// // verification in plaintext
	// float ver_x_times_y[nvals];
	// float ver_x_times_x[nvals];
	// float ver_y_times_y[nvals];
	// float ver_x_dot_y = 0;
	// float ver_norm_x = 0;
	// float ver_norm_y = 0;
	// std::cout << "values created" << std::endl;

	// // S_c(X,Y) = (X \dot Y) / (norm(X) * norm(Y))

	// for (uint32_t i = 0; i < nvals; i++)
	// {
	// 	float current_x = xembeddings[i];
	// 	float current_y = yembeddings[i];
	// 	float current_share = share_embeddings[i];
	// 	float current_share_prime = share_embeddings_prime[i];

	// 	uint32_t *xptr = (uint32_t *)&current_x;
	// 	uint32_t *yptr = (uint32_t *)&current_y;
	// 	uint32_t *shareptr = (uint32_t *)&current_share;
	// 	uint32_t *shareptr_prime = (uint32_t *)&current_share_prime;

	// 	xvals[i] = *xptr;
	// 	yvals[i] = *yptr;
	// 	sharevals[i] = *shareptr;
	// 	sharevals_prime[i] = *shareptr_prime;

	// 	ver_x_times_y[i] = current_x * current_y;
	// 	ver_x_dot_y += ver_x_times_y[i];

	// 	ver_x_times_x[i] = current_x * current_x;
	// 	ver_y_times_y[i] = current_y * current_y;
	// 	ver_norm_x += ver_x_times_x[i];
	// 	ver_norm_y += ver_y_times_y[i];
	// }

	// std::cout << "share vals: " << sharevals[0] << std::endl;
	// //std::cout << "Do we reach this part of the program?" << std::endl;
	// ver_norm_x = sqrt(ver_norm_x);
	// ver_norm_y = sqrt(ver_norm_y);

	// float ver_cos_sim = 1 - (ver_x_dot_y / (ver_norm_x * ver_norm_y));

	// std::cout << "cos_dist_ver: " << ver_cos_sim << std::endl;
	// // INPUTS
	// share *s_xin, *s_yin;
	// // // Input of the pre-computed shares of the face in the database
	// s_xin = ac->PutSharedSIMDINGate(nvals, sharevals_prime, bitlen);
	// s_yin = ac->PutSharedSIMDINGate(nvals, sharevals, bitlen);
	// // pairwise multiplication of all input values
	// share *s_out = BuildInnerProductCircuit(s_xin, s_yin, nvals,
	// 		(ArithmeticCircuit*) ac);

	// party->ExecCircuit();

	// std::cout << std::endl << "cos_dist_ver: " << ver_cos_sim << std::endl;

	// uint32_t output = s_out->get_clear_value<uint32_t>();

	// std::cout << "\nCircuit Result: " << output;
	// //std::cout << "\nVerification Result: " << v_sum << std::endl;
}

int main(int argc, char **argv)
{

	e_role role;
	uint32_t bitlen = 64, nvals = 128, secparam = 128, nthreads = 1;

	uint16_t port = 7766;
	std::string address = "127.0.0.1";
	int32_t test_op = -1;
	e_mt_gen_alg mt_alg = MT_OT;
	uint32_t test_bit = 0;
	uint32_t debug = 0;
	std::string inputfile;
	std::string pffrocd_path;

	read_test_options(&argc, &argv, &role, &bitlen, &nvals, &secparam, &address,
					  &port, &test_op, &test_bit, &mt_alg, &debug, &inputfile, &pffrocd_path);

	std::cout << std::fixed << std::setprecision(10);
	seclvl seclvl = get_sec_lvl(secparam);

	test_verilog_add64_SIMD(role, address, port, seclvl, nvals, nthreads, mt_alg, S_BOOL, debug, inputfile, pffrocd_path);

	return 0;
}

