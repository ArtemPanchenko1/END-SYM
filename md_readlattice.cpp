#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>

void ReadLattice(pNet_data &Pnet, potential_data& Po, char* filename1, char* filename2)
{
    double _1d_drr = 1.0 / Pnet.a_aver, rx, ry, rm, rmcut = Pnet.Rcut * Pnet.Rcut * RCC2;
    double* _s = (double*)malloc(2 * Pnet.SN * sizeof(double));
    unsigned int* _sc = (unsigned int*)malloc(2 * Pnet.ScN * sizeof(unsigned int));
    int* _sn = (int*)malloc(Pnet.ScN * sizeof(int));
    int i, j, k, m, readN;
    //double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
    std::ifstream file;
    //char filename[200];
    std::string d[4];
    //float d1[2];
    unsigned int d2, sn = Pnet.SN, scn = Pnet.ScN;
    file.open(filename1, std::ios::in);
    std::cerr << "Read file " << filename1 << "\n";
    //file << is << "\n";
    //file << "Step V FstartMinX FstartMaxX FstartMinY FstartMaxY FendMinX FendMaxX FendMinY FendMaxY\n";
    fprintf(stderr, "Start Read Data %i %i\n", Pnet.SN, Pnet.ScN);
    readN = 0;
    for (i = 0; i < Pnet.SN; ++i)
    {
        file >> d[0] ; file >> d[1]; file >> d[2];
#ifdef pre_ReadNodes3
        file >> d[3];
#endif // pre_ReadNodes3
        for (j = 0; j < 3; ++j)
            if (d[j].back() == ',') d[j].pop_back();
         
        _s[readN] = std::stod(d[1]) * ReadCoordinatesCoefficient;
        _s[readN + Pnet.SN] = std::stod(d[2]) * ReadCoordinatesCoefficient;
        rm = _s[readN] * _s[readN] + _s[readN + Pnet.SN] * _s[readN + Pnet.SN];
        if (rm < rmcut)
        {
            _sn[i] = readN;
            //if (i == 35544 || i== 43231 || i== 76827||i== 76828)
            //    std::wcerr << "T " << i << " " << readN << " " << _sn[i] << "\n";
            ++readN;
        } else
            _sn[i] = -1;         
        //std::cerr << i << " " << Pnet.h_S[i] << " " << Pnet.h_S[i + Pnet.SN] << "\n";
    }
    file.close();
    Pnet.SN = readN;
    fprintf(stderr, "Read %i nodes\n", readN);
    file.open(filename2, std::ios::in);
    //file << is << "\n";
    //file << "Step V FstartMinX FstartMaxX FstartMinY FstartMaxY FendMinX FendMaxX FendMinY FendMaxY\n";
    //fprintf(stderr, "Start Save Fmm %i\n",is);
    readN = 0;
    for (i = 0; i < Pnet.ScN; ++i)
    {
        file >> d[0]; file >> d[1]; file >> d[2];
        for (j = 0; j < 3; ++j)
            if (d[j].back() == ',') d[j].pop_back();

        _sc[readN] = std::stoi(d[1]) - 1;
        _sc[readN + Pnet.ScN] = std::stoi(d[2]) - 1;

        if (_sn[_sc[readN]] >= 0 && _sn[_sc[readN + Pnet.ScN]] >= 0)
        {
            /*if (readN == 17972 || readN == 18103)
            {
                std::cerr << i << " " << readN << " " << _sc[readN] << " " << _sc[readN + Pnet.ScN] <<" | "<< _sn[_sc[readN]]<<" "<< _sn[_sc[readN + Pnet.ScN]] << "\n";
                //std::cin.get();
            }/**/
            _sc[readN] = _sn[_sc[readN]];
            _sc[readN + Pnet.ScN] = _sn[_sc[readN + Pnet.ScN]];
            /*if (readN == 17972 || readN == 18103)
            {
                std::cerr << i << " " << readN << " " << _sc[readN] << " " << _sc[readN + Pnet.ScN] << "\n";
                //std::cin.get();
            }/**/
            ++readN;
        }
        
    }
    file.close();
    //std::cerr << "R1 " << _sc[17972] << " " << _sc[17972 + scn] << "\n";
    //std::cerr << "R2 " << _sc[18103] << " " << _sc[18103 + scn] << "\n";
    Pnet.ScN = readN;
    fprintf(stderr, "Read %i links\n", readN);
    //i--;std::cerr << i << " " << Pnet.h_Sc[i] << " " << Pnet.h_Sc[i + Pnet.ScN] << "\n";
    fprintf(stderr, "Finish Read Data\n", 0);
    //std::cin.get();
    //std::cerr << "B " << _s[0] << " " << _s[sn] << '\n';
#ifdef pre_deleteClose
    double* nox = (double*)malloc(Pnet.SN * sizeof(double));
    double* noy = (double*)malloc(Pnet.SN * sizeof(double));
    memcpy(nox, _s, Pnet.SN * sizeof(double));
    memcpy(noy, _s + sn, Pnet.SN * sizeof(double));
    unsigned int* nia = (unsigned int*)malloc(Pnet.ScN * sizeof(unsigned int));
    unsigned int* nib = (unsigned int*)malloc(Pnet.ScN * sizeof(unsigned int));
    memcpy(nia, _sc, Pnet.ScN * sizeof(unsigned int));
    memcpy(nib, _sc + scn, Pnet.ScN * sizeof(unsigned int));
    double _1d_drrmin = (4.0) / (Pnet.a_aver * Pnet.a_aver);
    //std::cerr << "TEST " << 1071 << " " << nia[1071] << " " << nib[1071] << "\n";
    //std::cerr << "TEST " << 6465 << " " << nia[6465] << " " << nib[6465] << "\n";
    //std::cerr << "TEST " << 14948 << " " << nia[14948] << " " << nib[14948] << "\n";
    //std::cerr << "TEST " << 15046 << " " << nia[15046] << " " << nib[15046] << "\n";
    uint_fast32_t deln, setn, repn;
    std::cerr << "N " << Pnet.SN << " " << Pnet.ScN << "\n";
    for (i = 0; i < Pnet.ScN; ++i)
    {
        rx = nox[nib[i]] - nox[nia[i]];
        ry = noy[nib[i]] - noy[nia[i]];
        rm = rx * rx + ry * ry;
        if (rm * _1d_drrmin < 1.0)
        {

            deln = nib[i];
            setn = nia[i];
            //std::cerr << "DELETE " << i << " " << setn << " " << deln << " | " << rm << " " << rm * _1d_drrmin << " " << _1d_drrmin << "\n";
            //std::cin.get();
            nia[i] = nia[Pnet.ScN - 1];
            nib[i] = nib[Pnet.ScN - 1];
            //std::cerr << "DELETE1 " << nia[i] << " " << nib[i] << "\n";
            --Pnet.ScN;
            for (j = 0; j < Pnet.ScN; ++j)
            {
                if (nia[j] == deln)
                {
                    nia[j] = setn;
                    //std::cerr << "REP1 " << j << " " << nia[j] << " " << nib[j] << "\n";
                }
                if (nib[j] == deln)
                {
                    nib[j] = setn;
                    //std::cerr << "REP2 " << j << " " << nia[j] << " " << nib[j] << "\n";
                }
            }
            nox[deln] = nox[Pnet.SN - 1];
            noy[deln] = noy[Pnet.SN - 1];
            repn = Pnet.SN - 1;
            --Pnet.SN;
            for (j = 0; j < Pnet.ScN; ++j)
            {
                if (nia[j] == repn)
                {
                    nia[j] = deln;
                    //std::cerr << "REP3 " << j << " " << nia[j] << " " << nib[j] << "\n";
                }
                if (nib[j] == repn)
                {
                    nib[j] = deln;
                    //std::cerr << "REP4 " << j << " " << nia[j] << " " << nib[j] << "\n";
                }
            }
            --i;
            //std::cerr << "\n";
        }
    }
    std::cerr << "N2 " << Pnet.SN << " " << Pnet.ScN << "\n";
    //std::cin.get();
    Pnet.h_S = (double*)malloc(2 * Pnet.SN * sizeof(double));
    memcpy(Pnet.h_S, nox, Pnet.SN * sizeof(double));
    memcpy(Pnet.h_S + Pnet.SN, noy, Pnet.SN * sizeof(double));
    Pnet.h_Sc = (unsigned int*)malloc(2 * Pnet.ScN * sizeof(unsigned int));
    memcpy(Pnet.h_Sc, nia, Pnet.ScN * sizeof(unsigned int));
    memcpy(Pnet.h_Sc + Pnet.ScN, nib, Pnet.ScN * sizeof(unsigned int));

    free(nox); nox = nullptr;
    free(noy); noy = nullptr;
    free(nia); nia = nullptr;
    free(nib); nib = nullptr;
#endif // pre_deleteClose
/*    for (i = 0; i < Pnet.ScN; ++i)
    {
        rx = nox[nib[i]] - nox[nia[i]];
        ry = noy[nib[i]] - noy[nia[i]];
        rm = rx * rx + ry * ry;
        if (rm * _1d_drrmin < 1.0)
        {

            deln = nib[i];
            setn = nia[i];
            std::cerr << "DELETE " << i << " " << setn << " " << deln << " | " << rm << " " << rm * _1d_drrmin << " " << _1d_drrmin << "\n";
            std::cin.get();
            Pnet.h_Sc[i] = Pnet.h_Sc[Pnet.ScN - 1];
            Pnet.h_Sc[i + Pnet.ScN] = Pnet.h_Sc[Pnet.ScN - 1 + Pnet.ScN];
            std::cerr << "DELETE1 " << Pnet.h_Sc[i] << " " << Pnet.h_Sc[i + Pnet.ScN] << "\n";
            --Pnet.ScN;
            for (j = 0; j < Pnet.ScN; ++j)
            {
                if (Pnet.h_Sc[j] == deln)
                {
                    Pnet.h_Sc[j] = setn;
                    std::cerr << "REP1 " << j << " " << Pnet.h_Sc[j] << " " << Pnet.h_Sc[j + Pnet.ScN] << "\n";
                }
                else if (Pnet.h_Sc[j + Pnet.ScN] == deln)
                {
                    Pnet.h_Sc[j + Pnet.ScN] = setn;
                    std::cerr << "REP2 " << j << " " << Pnet.h_Sc[j] << " " << Pnet.h_Sc[j + Pnet.ScN] << "\n";
                }
            }
            Pnet.h_S[deln] = Pnet.h_S[Pnet.SN - 1];
            Pnet.h_S[deln + Pnet.SN] = Pnet.h_S[Pnet.SN - 1 + Pnet.SN];
            repn = Pnet.SN - 1;
            --Pnet.SN;
            for (j = 0; j < Pnet.ScN; ++j)
            {
                if (Pnet.h_Sc[j] == repn)
                {
                    Pnet.h_Sc[j] = deln;
                    std::cerr << "REP3 " << j << " " << Pnet.h_Sc[j] << " " << Pnet.h_Sc[j + Pnet.ScN] << "\n";
                }
                else if (Pnet.h_Sc[j + Pnet.ScN] == repn)
                {
                    Pnet.h_Sc[j + Pnet.ScN] = deln;
                    std::cerr << "REP4 " << j << " " << Pnet.h_Sc[j] << " " << Pnet.h_Sc[j + Pnet.ScN] << "\n";
                }
            }
            --i;
        }
    }
    std::cerr << "N2 " << Pnet.SN << " " << Pnet.ScN << "\n";
    std::cin.get();
#endif // pre_deleteClose/**/
    //std::cerr << "R1 " << _sc[17972] << " " << _sc[17972 + scn] << "\n";
    //std::cerr << "R2 " << _sc[18103] << " " << _sc[18103 + scn] << "\n";
#ifndef pre_deleteClose
    Pnet.h_S = (double*)malloc(2 * Pnet.SN * sizeof(double));
    memcpy(Pnet.h_S, _s, Pnet.SN * sizeof(double));
    memcpy(Pnet.h_S + Pnet.SN, _s + sn, Pnet.SN * sizeof(double));
    Pnet.h_Sc = (unsigned int*)malloc(2 * Pnet.ScN * sizeof(unsigned int));
    memcpy(Pnet.h_Sc, _sc, Pnet.ScN * sizeof(unsigned int));
    memcpy(Pnet.h_Sc + Pnet.ScN, _sc + scn, Pnet.ScN * sizeof(unsigned int));
#endif // !pre_deleteClose

    //std::cerr << "E1 " <<  Pnet.h_Sc[17972] << " " << Pnet.h_Sc[17972 + Pnet.ScN] << "\n";
    //std::cerr << "E2 " << Pnet.h_Sc[18103] << " " << Pnet.h_Sc[18103 + Pnet.ScN] << "\n";
#ifdef pre_GetNetworkParameters
    unsigned int* nodelinks = (unsigned int*)malloc(Pnet.SN * sizeof(unsigned int));
    memset(nodelinks, 0, Pnet.SN * sizeof(unsigned int));
    double inodex, inodey, jnodex, jnodey, irr, jrr, rrout = 0.9 * Pnet.Rcut * Pnet.Rcut * RCC2, R0 = 0.03987475 * RCC,
        rrin = (3.0 + Pnet.CellDistance) * (3.0 + Pnet.CellDistance) * R0 * R0;
    int inode, jnode;
    std::cerr << "R param " << sqrt(rrout) << " " << sqrt(rrin) << "\n";
    double nlinks = 0, mllinks = 0, mlllinks = 0, rlink, rcx, rcy, rrc, nrc = 0, nrcarea, slc = 0;
    bool iin, jin;
    for (i = 0; i < Pnet.ScN; ++i)
    {
        iin = false;
        jin = false;
        inode = Pnet.h_Sc[i];
        jnode = Pnet.h_Sc[i + Pnet.ScN];
        inodex = Pnet.h_S[inode];
        inodey = Pnet.h_S[inode + Pnet.SN];
        jnodex = Pnet.h_S[jnode];
        jnodey = Pnet.h_S[jnode + Pnet.SN];
        irr = inodex * inodex + inodey * inodey;
        jrr = jnodex * jnodex + jnodey * jnodey;

        rcx = 0.5 * (inodex + inodex);
        rcy = 0.5 * (inodey + inodey);
        rrc = rcx * rcx + rcy * rcy;

        if (irr > rrin && irr < rrout)
        {
            ++nodelinks[inode];
            iin = true;

        }
        if (jrr > rrin && jrr < rrout)
        {
            ++nodelinks[jnode];
            jin = true;
        }
        if (iin || jin)
        {
            nlinks += 1;
            rlink = sqrt((jnodex - inodex) * (jnodex - inodex) + (jnodey - inodey) * (jnodey - inodey));
            mllinks += rlink;
            mlllinks += rlink * rlink;
        }
        if (rrc > rrin && rrc < rrout)
        {
            nrc += 1;
            slc += sqrt((jnodex - inodex) * (jnodex - inodex) + (jnodey - inodey) * (jnodey - inodey));
        }
    }
    mllinks /= nlinks;
    mlllinks /= nlinks;
    nrcarea = nrc / (MC_pi * (rrout - rrin));
    double nnodes = 0, mnnodes = 0, mnnnodes = 0;
    for (i = 0; i < Pnet.SN; ++i)
    {
        if (nodelinks[i] > 0)
        {
            mnnodes += nodelinks[i];
            mnnnodes += nodelinks[i] * nodelinks[i];
            nnodes += 1;
        }
        if (nodelinks[i] < 3 && nodelinks[i] > 0)
            std::cerr << "LESS " << i << " " << nodelinks[i] << "\n";
    }
    mnnodes /= double(nnodes);
    mnnnodes /= double(nnodes);
    std::cerr << "Condactivity " << nnodes << " " << mnnodes << " " << sqrt(mnnnodes - mnnodes * mnnodes) << "\n";
    std::cerr << "Linklength " << nlinks << " " << mllinks*length_const << " " << sqrt(mlllinks - mllinks * mllinks) * length_const
        << " | " << mllinks / (4 * R0) << " " << sqrt(mlllinks - mllinks * mllinks) / (4 * R0) << "\n";
    std::cerr << "Link in area " << nrc << " " << nrcarea / (length_const * length_const) << " " << nrcarea * mllinks * mllinks
        << "\nPore " << (MC_pi * (rrout - rrin) - slc * 2e-4 * RCC) * (length_const * length_const)
        << " " << (MC_pi * (rrout - rrin) - slc * 2e-4 * RCC) * (mllinks * mllinks) / (MC_pi * (rrout - rrin))
        << " " << (1 - 2e-4 * RCC * mllinks * nrcarea) * mllinks * mllinks << "\n";
    std::cerr << "AA " << 2e-4 * RCC * mllinks * nrcarea << "\n";
    Po.hfreefiber = (0.5 * MC_pi * (rrout - rrin) / slc);// (0.5 * MC_pi * (rrout - rrin) / slc - Po.rfiber);
    std::cerr << "Free area " << MC_pi * (rrout - rrin) / slc * length_const << " | " << (0.5 * MC_pi * (rrout - rrin) / slc - Po.rfiber) * length_const << " | " << 1.0 / (double(0.5 * Po.hfreefiber) * log(0.5 * double(Po.hfreefiber) / double(Po.rfiber))) << "\n";
   
    std::cin.get();
    free(nodelinks);
#endif // pre_GetNetworkParameters


    //std::cerr << "B " << Pnet.h_S[0] << " " << Pnet.h_S[Pnet.SN] << '\n';
    int adn;
    Pnet.L = 0; Pnet.minrm = 1e20; Pnet.maxrm = -1e20;
    Pnet.AdN = 0;
    for (i = 0; i < Pnet.ScN; ++i)
    {
        rx = Pnet.h_S[Pnet.h_Sc[i + Pnet.ScN]] - Pnet.h_S[Pnet.h_Sc[i]];
        ry = Pnet.h_S[Pnet.h_Sc[i + Pnet.ScN] + Pnet.SN] - Pnet.h_S[Pnet.h_Sc[i] + Pnet.SN];
        rm = sqrt(rx * rx + ry * ry);
        /*if (rm > 1e-1)
        {
            std::cerr << "ERR1 " <<i<<" " << rm << " " << Pnet.h_Sc[i] << " " << Pnet.h_Sc[i + Pnet.ScN] << "\n";
            std::cin.get();
        }/**/
        adn = int(rm * _1d_drr) - 1;
        if (adn < 0) adn = 0;
        Pnet.AdN += adn;
        if (Pnet.minrm > rm)
        {
            Pnet.minrm = rm;
            //std::cerr << minrm << " " << rm << " | " << rx <<" "<< ry << "\n";
            //std::cin.get();
        }
        if (Pnet.maxrm < rm)
        {
            Pnet.maxrm = rm;
            //std::cerr << maxrm << " " << rm << " | " << rx <<" "<< ry << "\n";
            //std::cin.get();
        }
        Pnet.L += rm;
    }

    std::cerr << Pnet.L << " " << Pnet.minrm << " " << Pnet.maxrm << " | " << Pnet.AdN << "\n";
    std::cerr << "Add N " << Pnet.SN << " " << Pnet.ScN << " " << Pnet.AdN << "\n";
    double *h_S_new = (double*)malloc(2 * ( Pnet.SN + Pnet.AdN ) * sizeof(double));
    unsigned int *h_Sc_new = (unsigned int*)malloc(2 * ( Pnet.ScN + Pnet.AdN ) * sizeof(unsigned int));
    memcpy(h_S_new, Pnet.h_S, Pnet.SN * sizeof(double));
    memcpy(h_S_new + Pnet.SN + Pnet.AdN, Pnet.h_S + Pnet.SN, Pnet.SN * sizeof(double));
    memcpy(h_Sc_new, Pnet.h_Sc, Pnet.ScN * sizeof(unsigned int));
    memcpy(h_Sc_new + Pnet.ScN + Pnet.AdN, Pnet.h_Sc + Pnet.ScN, Pnet.ScN * sizeof(unsigned int));
    double* h_S_t = Pnet.h_S;
    unsigned int* h_Sc_t = Pnet.h_Sc;
    Pnet.h_S = h_S_new;
    Pnet.h_Sc = h_Sc_new;
    free(h_S_t); h_S_t = nullptr; h_S_new = nullptr;
    free(h_Sc_t); h_Sc_t = nullptr; h_Sc_new = nullptr;

    double _1d_rm, dr, drm;
    unsigned int Adn = 0;
    for (k = 0; k < Pnet.ScN; ++k)
    {
        i = Pnet.h_Sc[k];
        j = Pnet.h_Sc[k + Pnet.ScN + Pnet.AdN];
        rx = Pnet.h_S[j] - Pnet.h_S[i];
        ry = Pnet.h_S[j + Pnet.SN + Pnet.AdN] - Pnet.h_S[i + Pnet.SN + Pnet.AdN];
        rm = sqrt(rx * rx + ry * ry);
        adn = int(rm * _1d_drr) - 1;
        if (adn <= 0)
        {
            continue;
        }
        _1d_rm = 1.0 / rm;
        dr = rm / double( adn + 1 );
        Pnet.h_Sc[k + Pnet.ScN + Pnet.AdN] = Pnet.SN + Adn;
        //std::cerr << "change " << k << " | "<< Pnet.h_Sc[k] << " " << Pnet.h_Sc[k + Pnet.ScN] <<" | " <<j << "\n";
        for (m = 0; m < adn; ++m)
        {
            drm = (m + 1) * dr;
            Pnet.h_S[Pnet.SN + Adn] = drm * rx * _1d_rm + Pnet.h_S[i];
            Pnet.h_S[2 * Pnet.SN + Pnet.AdN + Adn] = drm * ry * _1d_rm + Pnet.h_S[i + Pnet.SN + Pnet.AdN];
            if (m + 1 < adn)
            {
                Pnet.h_Sc[Pnet.ScN + Adn] = Pnet.SN + Adn;
                Pnet.h_Sc[2 * Pnet.ScN + Pnet.AdN + Adn] = Pnet.SN + Adn + 1;
                //std::cerr << "add " << Pnet.h_Sc[Pnet.ScN + Adn] <<" "<< Pnet.h_Sc[2 * Pnet.ScN + Pnet.AdN + Adn] << "\n";
            }
            else
            {
                Pnet.h_Sc[Pnet.ScN + Adn] = Pnet.SN + Adn;
                Pnet.h_Sc[2 * Pnet.ScN + Pnet.AdN + Adn] = j;
                //std::cerr << "add " << Pnet.h_Sc[Pnet.ScN + Adn] << " " << Pnet.h_Sc[2 * Pnet.ScN + Pnet.AdN + Adn] << "\n";
            }
            //std::cerr << "A " << m << " " << adn << " " << Adn << " | " << Pnet.h_S[Pnet.SN + Adn] << " " << Pnet.h_S[2 * Pnet.SN + Pnet.AdN + Adn]
            //    <<" | "<< drm << " " << rx << " " << ry << " " << rm << "\n";
            
            ++Adn;
        }   
        //std::cin.get();
    }
    Pnet.SN += Pnet.AdN;
    Pnet.ScN += Pnet.AdN;

    free(_sn);
    _sn = nullptr;
    free(_s);
    _s = nullptr;
    free(_sc);
    _sc = nullptr;
    
    //std::cin.get();
    //fprintf(stderr, "Finish Save Fmm\n", 0);    
}


