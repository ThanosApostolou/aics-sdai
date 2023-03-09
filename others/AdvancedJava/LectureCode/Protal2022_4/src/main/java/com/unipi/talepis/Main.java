package com.unipi.talepis;

import com.google.gson.GsonBuilder;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class Main {
    public static List<Block> blockChain = new ArrayList<>();
    public static int prefix = 4;
    public static void main(String[] args) {
        System.out.println("Process started");
        //1st Block
        Block genesisBlock = new Block("0","Data for the first block",
                new Date().getTime());
        genesisBlock.mineBlock(prefix);
        blockChain.add(genesisBlock);
        System.out.println("Node "+(blockChain.size())+" created!");

        //2nd Block
        Block secondBlock = new Block(blockChain.get(blockChain.size()-1).getHash(),
                "Data for the second block",
                new Date().getTime());
        secondBlock.mineBlock(prefix);
        blockChain.add(secondBlock);
        System.out.println("Node "+(blockChain.size())+" created!");
        //3rd Block
        Block thirdBlock = new Block(blockChain.get(blockChain.size()-1).getHash(),
                "Data for the third block",
                new Date().getTime());
        thirdBlock.mineBlock(prefix);
        blockChain.add(thirdBlock);
        System.out.println("Node "+(blockChain.size())+" created!");

        //Transform into Json
        String json = new GsonBuilder().setPrettyPrinting().create().toJson(blockChain);
        System.out.println("The blockChain:");
        System.out.println(json);

        //Validate BlockChain
        System.out.println("Is chain valid?:"+isChainValid());
    }
    public static boolean isChainValid(){
        Block currentBlock;
        Block previousBlock;
        String hashTarget = new String(new char[prefix]).replace('\0','0');
        for (int i=1;i<blockChain.size();i++){
            currentBlock = blockChain.get(i);
            previousBlock = blockChain.get(i-1);
            if (!currentBlock.getHash().equals(currentBlock.calculateBlockHash()))
                return false;
            if (!previousBlock.getHash().equals(currentBlock.getPreviousHash()))
                return false;
            if (!currentBlock.getHash().substring(0,prefix).equals(hashTarget))
                return false;
        }
        return true;
    }
}