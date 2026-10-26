//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

/*
 * Generated from doc/logo.png by scripts/generate-splash-logo,
 * do not edit.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace ryujin::splash
{
  /*
   * The logo in 49x64 pixels, one character of [0-9a-zA-Z] per
   * pixel indexing the palette, and '.' for a transparent pixel:
   */

  // clang-format off
  constexpr std::array<std::uint32_t, 58> logo_palette = {
    0x686758, 0xa3966e, 0x9b9b89, 0xe1d1a4, 0x87715e, 0x54473a,
    0x162424, 0x974a38, 0x788376, 0x97301a, 0xc5b17e, 0x3e524d,
    0x5e261a, 0xd3bd89, 0xbbc0ad, 0x353e40, 0xc67158, 0x293533,
    0x222c2b, 0x999785, 0x1c2827, 0x24312d, 0xa9a894, 0xa6a38a,
    0x323c38, 0x898877, 0x3a4743, 0xb6b399, 0x36423d, 0x46524b,
    0x96927a, 0x777768, 0xa29d87, 0x56645a, 0xb2ad95, 0x4a5b53,
    0xb5ab87, 0x434a44, 0x928d79, 0xc6bb96, 0xd7c698, 0xe7d5a8,
    0x565b53, 0x66756a, 0xc5b688, 0xa69a77, 0x666757, 0x55574a,
    0xb7a87a, 0x6a6d63, 0x7a7d72, 0xbab9a3, 0x878267, 0xd8cba3,
    0xc8c1a5, 0xe1cc99, 0x242e30, 0xc3bda4,
  };

  constexpr std::array<std::string_view, 64> logo_pixels = {
    "nnwyjnwDMTJ1x....................................",
    "pmjrjyMFuJ0Lz....................................",
    "nDymr3MaIKtt.....................................",
    "yCwnD3FIKtx......................................",
    "mCnDEJM1ho.......................................",
    "uSrDJE0ot........................................",
    "uuA3IQxh.........................................",
    "Ay31EKs..........................................",
    "ny3T0B...........................................",
    "myAM5i...........................................",
    "jA3Lk............................................",
    "yD4ol............................................",
    "nDLk.............................................",
    "nKh..............................................",
    "QBk.........................kik..................",
    "uo.....................k66.koNNo.................",
    "Lh....................kBxtkkvVjG.................",
    "vh...........tqki..lBotmrnNj2Dw6.......kkk.......",
    "5h...........HjvohkHPrmmmnmjVDpb..llokkoGGl......",
    ".......thhhUkklOjHGvmm2ywjppw0w26iBjuuyCunv0.....",
    ".......2mjvKNGtppjjnrrymjjp2jCCCttvjnRS4pJup.....",
    "......qoho8mjnmrnpmmnjmmmSwpCJjpjmpwRwmQCu5k.....",
    "......NOHN2j2nwmjnrmjrSSRD4e04yjvjjyyppwHh6......",
    "......fG2jjpjjmjrPPjj3FFVH4rjjnmjjwmnmjjc99isBok.",
    ".....kNppjjjjpmmrrPjmVmr0K8wwjjn41pSwjr4cg90JwLi.",
    ".vNGotppjmmm2mCKjnmjCjpjwm2yr447790VnmA797770jjL.",
    ".2mmmj2pm24pPwpjppjjjppjj00m44vv8p4rjm199940jjvL.",
    ".UoOm2P2jmm8pjmjQJAjpjmw45c50BohlOtwpp44774m2OBh.",
    "..tvmwyjjjuCwuu41nnp2jm773ujGi....ijp2vC45yjoho..",
    "..iGrjmmrpnDDAJAAj2ppwm50ySvi.....UvjjmnCypGq....",
    "phq22mrmjn1upnSjpPjjmwVwjpKth.....hsjuCj2mBk.....",
    "phxjpmpmmjmA11IAE0jppjjmppoh.......kovppOv6......",
    "posBnpPmjymmryARIIACnnj2BOh.........itvOtBk......",
    "nNiHmm2jSpVm2jnnRDjrnrjsh............hoOB........",
    "mplGNemmjjCmjPVrJ03jjCnpohh......................",
    "2wKkiGm2jjmjVjjDjnjC3pwDEvoh.....................",
    "rmplhivsv2p0juVvmmuFRQDDnAIUi....................",
    "jPjBh....tx2jjyjjyyjn3AnA3n4oi...................",
    "jmSjl.....khjmOrjypmnIFFEnJEvqh..................",
    "j0wPsh.....ioxzjjmjmrnAIDDDC3Jko........hkil.....",
    "pjpmvHql......ljvpjppwD3nnMERCJs.......hsvvBkh...",
    "nrmOmrOl.......xlvwjruu1TEDAudFti......hGCjpto...",
    "mumujmrtk.......hBjuCjDEAMDI33AQs.iiillhxnDjpj...",
    "3uPwmpjmoi.......oBvjmmnFFArru1TLkovGG5LjnBLu8...",
    "nmj4jnjmjGx......liKnjnrr1IIIa325hvCjrppps6tm8k..",
    "yjVurumjj22Niikl.LunwrnrEFRDFFA1AsCjj8uCskkL3vi..",
    "jymuuujnjjjjvNOsstvpmpwn3A1I2MIFJiqjAup4o.lAnii..",
    "nymrjjpwnmppvmrmppjwmuurAnAADRDJIilppnppi.sKlil..",
    "mjnCpjrnujCjpju2njjnpnCrn1MEM11TIhtyujjvlhhBQ0lk.",
    "jrrurjpmmnj4yjjwnmujpwnyETTF3FFEQvKApun5sBxpmpxB.",
    "rnmywurjnpmvrjjjjnnnnjjrA11MJ1MQvvj0vQCvpjmPmjPH.",
    "ujprnyrjPQj2vjjjjjwujjrRF33FEIa4NpnpjjmjCujvQOOo.",
    "jDwjnnjjnnnrwnwrunrjCrF311J11JushnFpmvsoooisuwlh.",
    "nCnrnQnmnnynnyjwjmmPyRIFTAITEEvsinnqohs..htnrth..",
    "DnpmmnuLpnnymmmjjnnAEFMu2DA110ihohhh.....o0uBk...",
    "CnnwDrvKHuJFE3SIRdEF1M3EDDaTIti..........lkUh....",
    "nnnrrmpvQ3MEJ13dA31R3a111Iw0oi...................",
    "yrjprruvvpjCu1RF1TE12EEMMQsih....................",
    "SyjjrrrvG501F31ETJIEM1K0Koh......................",
    "unnnAruvNKoo045511L00Bohho.......................",
    "uprrvjvNKs.iilikihhh.............................",
    "rruvoGxot........................................",
    "KGoli............................................",
    "hhh..............................................",
  };
  // clang-format on
} // namespace ryujin::splash
