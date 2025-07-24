# -*- coding: utf-8 -*-
"""
区块链和Web3集成模块
提供智能合约交互、加密货币钱包、DeFi协议、NFT等功能
"""

import asyncio
import json
import time
import hashlib
import secrets
import base64
from typing import (
    Dict, List, Any, Optional, Union, Tuple, Callable,
    Set, AsyncIterator, Iterator
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from decimal import Decimal
import re

# Web3和以太坊库
from web3 import Web3, HTTPProvider, WebsocketProvider
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_keys import keys
from eth_utils import to_checksum_address, is_address
from eth_typing import Address, HexStr

# 智能合约编译
from solcx import compile_source, install_solc, set_solc_version

# 加密和安全
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import mnemonic

# IPFS集成
import ipfshttpclient

# 数据库
import sqlite3
import aiosqlite
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# HTTP客户端
import aiohttp
from aiohttp import ClientSession, ClientTimeout

# 配置和日志
import structlog
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# 其他工具
import qrcode
from PIL import Image
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


class BlockchainNetwork(Enum):
    """区块链网络枚举"""
    ETHEREUM_MAINNET = "ethereum_mainnet"
    ETHEREUM_GOERLI = "ethereum_goerli"
    ETHEREUM_SEPOLIA = "ethereum_sepolia"
    POLYGON_MAINNET = "polygon_mainnet"
    POLYGON_MUMBAI = "polygon_mumbai"
    BSC_MAINNET = "bsc_mainnet"
    BSC_TESTNET = "bsc_testnet"
    ARBITRUM_MAINNET = "arbitrum_mainnet"
    OPTIMISM_MAINNET = "optimism_mainnet"
    AVALANCHE_MAINNET = "avalanche_mainnet"
    FANTOM_MAINNET = "fantom_mainnet"
    LOCAL_GANACHE = "local_ganache"


class TransactionStatus(Enum):
    """交易状态枚举"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVERTED = "reverted"
    DROPPED = "dropped"


class TokenStandard(Enum):
    """代币标准枚举"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    BEP20 = "bep20"
    SPL = "spl"


class WalletType(Enum):
    """钱包类型枚举"""
    HOT_WALLET = "hot_wallet"
    COLD_WALLET = "cold_wallet"
    HARDWARE_WALLET = "hardware_wallet"
    MULTI_SIG = "multi_sig"
    SMART_CONTRACT = "smart_contract"


@dataclass
class NetworkConfig:
    """网络配置"""
    name: str
    network: BlockchainNetwork
    rpc_url: str
    chain_id: int
    currency_symbol: str
    block_explorer_url: str
    gas_price_api: Optional[str] = None
    is_testnet: bool = False
    supports_eip1559: bool = True


@dataclass
class TokenInfo:
    """代币信息"""
    address: str
    symbol: str
    name: str
    decimals: int
    standard: TokenStandard
    total_supply: Optional[int] = None
    logo_url: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TransactionInfo:
    """交易信息"""
    hash: str
    from_address: str
    to_address: str
    value: Decimal
    gas_price: int
    gas_limit: int
    gas_used: Optional[int] = None
    status: TransactionStatus = TransactionStatus.PENDING
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    transaction_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    input_data: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class WalletInfo:
    """钱包信息"""
    address: str
    private_key: Optional[str] = None
    public_key: Optional[str] = None
    mnemonic: Optional[str] = None
    wallet_type: WalletType = WalletType.HOT_WALLET
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    is_encrypted: bool = False
    derivation_path: Optional[str] = None


@dataclass
class NFTMetadata:
    """NFT元数据"""
    name: str
    description: str
    image: str
    external_url: Optional[str] = None
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    background_color: Optional[str] = None
    animation_url: Optional[str] = None
    youtube_url: Optional[str] = None


@dataclass
class NFTInfo:
    """NFT信息"""
    contract_address: str
    token_id: int
    owner: str
    metadata: NFTMetadata
    token_uri: Optional[str] = None
    standard: TokenStandard = TokenStandard.ERC721
    collection_name: Optional[str] = None
    rarity_rank: Optional[int] = None
    last_sale_price: Optional[Decimal] = None
    current_price: Optional[Decimal] = None


class Web3Provider:
    """Web3提供者"""
    
    def __init__(self, network_config: NetworkConfig):
        self.config = network_config
        self.w3 = Web3(HTTPProvider(network_config.rpc_url))
        
        # 添加POA中间件（用于某些网络）
        if network_config.network in [BlockchainNetwork.BSC_MAINNET, BlockchainNetwork.BSC_TESTNET]:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.is_connected = self.w3.isConnected()
        
        if not self.is_connected:
            logger.warning(f"无法连接到网络: {network_config.name}")
    
    def get_balance(self, address: str) -> Decimal:
        """获取地址余额"""
        try:
            balance_wei = self.w3.eth.get_balance(to_checksum_address(address))
            return Decimal(self.w3.fromWei(balance_wei, 'ether'))
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return Decimal('0')
    
    def get_transaction(self, tx_hash: str) -> Optional[TransactionInfo]:
        """获取交易信息"""
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = None
            
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            except:
                pass
            
            status = TransactionStatus.PENDING
            if receipt:
                if receipt.status == 1:
                    status = TransactionStatus.CONFIRMED
                else:
                    status = TransactionStatus.FAILED
            
            return TransactionInfo(
                hash=tx_hash,
                from_address=tx['from'],
                to_address=tx['to'] or '',
                value=Decimal(self.w3.fromWei(tx['value'], 'ether')),
                gas_price=tx['gasPrice'],
                gas_limit=tx['gas'],
                gas_used=receipt.gasUsed if receipt else None,
                status=status,
                block_number=tx.get('blockNumber'),
                block_hash=tx.get('blockHash'),
                transaction_index=tx.get('transactionIndex'),
                input_data=tx.get('input'),
                logs=receipt.logs if receipt else []
            )
        
        except Exception as e:
            logger.error(f"获取交易信息失败: {e}")
            return None
    
    def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """估算Gas费用"""
        try:
            return self.w3.eth.estimate_gas(transaction)
        except Exception as e:
            logger.error(f"估算Gas失败: {e}")
            return 21000  # 默认值
    
    def get_gas_price(self) -> int:
        """获取Gas价格"""
        try:
            return self.w3.eth.gas_price
        except Exception as e:
            logger.error(f"获取Gas价格失败: {e}")
            return self.w3.toWei('20', 'gwei')  # 默认值
    
    def get_nonce(self, address: str) -> int:
        """获取nonce"""
        try:
            return self.w3.eth.get_transaction_count(to_checksum_address(address))
        except Exception as e:
            logger.error(f"获取nonce失败: {e}")
            return 0
    
    def send_transaction(self, signed_tx: HexStr) -> str:
        """发送交易"""
        try:
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx)
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"发送交易失败: {e}")
            raise
    
    def wait_for_transaction(self, tx_hash: str, timeout: int = 120) -> Optional[Dict[str, Any]]:
        """等待交易确认"""
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            return dict(receipt)
        except Exception as e:
            logger.error(f"等待交易确认失败: {e}")
            return None


class WalletManager:
    """钱包管理器"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.wallets: Dict[str, WalletInfo] = {}
        self.mnemonic_generator = mnemonic.Mnemonic("english")
    
    def create_wallet(self, name: Optional[str] = None, 
                     password: Optional[str] = None) -> WalletInfo:
        """创建新钱包"""
        # 生成助记词
        mnemonic_phrase = self.mnemonic_generator.generate(strength=128)
        
        # 从助记词生成账户
        Account.enable_unaudited_hdwallet_features()
        account = Account.from_mnemonic(mnemonic_phrase)
        
        # 创建钱包信息
        wallet = WalletInfo(
            address=account.address,
            private_key=account.privateKey.hex(),
            mnemonic=mnemonic_phrase,
            name=name or f"Wallet_{len(self.wallets) + 1}",
            derivation_path="m/44'/60'/0'/0/0"
        )
        
        # 如果提供了密码，加密私钥和助记词
        if password:
            wallet.private_key = self._encrypt_data(wallet.private_key, password)
            wallet.mnemonic = self._encrypt_data(wallet.mnemonic, password)
            wallet.is_encrypted = True
        
        self.wallets[wallet.address] = wallet
        logger.info(f"钱包已创建: {wallet.address}")
        
        return wallet
    
    def import_wallet_from_private_key(self, private_key: str, 
                                     name: Optional[str] = None,
                                     password: Optional[str] = None) -> WalletInfo:
        """从私钥导入钱包"""
        try:
            account = Account.from_key(private_key)
            
            wallet = WalletInfo(
                address=account.address,
                private_key=private_key,
                name=name or f"Imported_{len(self.wallets) + 1}"
            )
            
            if password:
                wallet.private_key = self._encrypt_data(wallet.private_key, password)
                wallet.is_encrypted = True
            
            self.wallets[wallet.address] = wallet
            logger.info(f"钱包已导入: {wallet.address}")
            
            return wallet
        
        except Exception as e:
            logger.error(f"导入钱包失败: {e}")
            raise
    
    def import_wallet_from_mnemonic(self, mnemonic_phrase: str,
                                  derivation_path: str = "m/44'/60'/0'/0/0",
                                  name: Optional[str] = None,
                                  password: Optional[str] = None) -> WalletInfo:
        """从助记词导入钱包"""
        try:
            # 验证助记词
            if not self.mnemonic_generator.check(mnemonic_phrase):
                raise ValueError("无效的助记词")
            
            Account.enable_unaudited_hdwallet_features()
            account = Account.from_mnemonic(mnemonic_phrase, passphrase="", account_path=derivation_path)
            
            wallet = WalletInfo(
                address=account.address,
                private_key=account.privateKey.hex(),
                mnemonic=mnemonic_phrase,
                name=name or f"Mnemonic_{len(self.wallets) + 1}",
                derivation_path=derivation_path
            )
            
            if password:
                wallet.private_key = self._encrypt_data(wallet.private_key, password)
                wallet.mnemonic = self._encrypt_data(wallet.mnemonic, password)
                wallet.is_encrypted = True
            
            self.wallets[wallet.address] = wallet
            logger.info(f"钱包已从助记词导入: {wallet.address}")
            
            return wallet
        
        except Exception as e:
            logger.error(f"从助记词导入钱包失败: {e}")
            raise
    
    def get_wallet(self, address: str) -> Optional[WalletInfo]:
        """获取钱包"""
        return self.wallets.get(to_checksum_address(address))
    
    def list_wallets(self) -> List[WalletInfo]:
        """列出所有钱包"""
        return list(self.wallets.values())
    
    def delete_wallet(self, address: str) -> bool:
        """删除钱包"""
        try:
            address = to_checksum_address(address)
            if address in self.wallets:
                del self.wallets[address]
                logger.info(f"钱包已删除: {address}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除钱包失败: {e}")
            return False
    
    def decrypt_private_key(self, wallet: WalletInfo, password: str) -> str:
        """解密私钥"""
        if not wallet.is_encrypted:
            return wallet.private_key
        
        try:
            return self._decrypt_data(wallet.private_key, password)
        except Exception as e:
            logger.error(f"解密私钥失败: {e}")
            raise ValueError("密码错误或解密失败")
    
    def decrypt_mnemonic(self, wallet: WalletInfo, password: str) -> str:
        """解密助记词"""
        if not wallet.is_encrypted or not wallet.mnemonic:
            return wallet.mnemonic
        
        try:
            return self._decrypt_data(wallet.mnemonic, password)
        except Exception as e:
            logger.error(f"解密助记词失败: {e}")
            raise ValueError("密码错误或解密失败")
    
    def generate_qr_code(self, address: str, amount: Optional[Decimal] = None) -> Image.Image:
        """生成钱包地址二维码"""
        data = address
        if amount:
            data = f"ethereum:{address}?value={amount}"
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        return qr.make_image(fill_color="black", back_color="white")
    
    def _encrypt_data(self, data: str, password: str) -> str:
        """加密数据"""
        # 使用密码派生密钥
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        cipher = Fernet(key)
        
        encrypted_data = cipher.encrypt(data.encode())
        return base64.b64encode(salt + encrypted_data).decode()
    
    def _decrypt_data(self, encrypted_data: str, password: str) -> str:
        """解密数据"""
        data = base64.b64decode(encrypted_data.encode())
        salt = data[:16]
        encrypted = data[16:]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        cipher = Fernet(key)
        
        decrypted_data = cipher.decrypt(encrypted)
        return decrypted_data.decode()


class SmartContractManager:
    """智能合约管理器"""
    
    def __init__(self, web3_provider: Web3Provider):
        self.provider = web3_provider
        self.w3 = web3_provider.w3
        self.contracts: Dict[str, Any] = {}
    
    def compile_contract(self, source_code: str, contract_name: str) -> Dict[str, Any]:
        """编译智能合约"""
        try:
            # 安装和设置Solidity编译器版本
            install_solc('0.8.19')
            set_solc_version('0.8.19')
            
            compiled_sol = compile_source(source_code)
            contract_interface = compiled_sol[f'<stdin>:{contract_name}']
            
            return {
                'abi': contract_interface['abi'],
                'bytecode': contract_interface['bin'],
                'source': source_code
            }
        
        except Exception as e:
            logger.error(f"编译合约失败: {e}")
            raise
    
    def deploy_contract(self, contract_info: Dict[str, Any], 
                       constructor_args: List[Any],
                       wallet: WalletInfo,
                       password: Optional[str] = None) -> str:
        """部署智能合约"""
        try:
            # 获取私钥
            if wallet.is_encrypted and password:
                private_key = self._decrypt_private_key(wallet, password)
            else:
                private_key = wallet.private_key
            
            account = Account.from_key(private_key)
            
            # 创建合约对象
            contract = self.w3.eth.contract(
                abi=contract_info['abi'],
                bytecode=contract_info['bytecode']
            )
            
            # 构建部署交易
            constructor = contract.constructor(*constructor_args)
            
            # 估算Gas
            gas_estimate = constructor.estimateGas({
                'from': account.address
            })
            
            # 构建交易
            transaction = constructor.buildTransaction({
                'from': account.address,
                'gas': gas_estimate,
                'gasPrice': self.provider.get_gas_price(),
                'nonce': self.provider.get_nonce(account.address)
            })
            
            # 签名交易
            signed_txn = account.sign_transaction(transaction)
            
            # 发送交易
            tx_hash = self.provider.send_transaction(signed_txn.rawTransaction)
            
            # 等待交易确认
            receipt = self.provider.wait_for_transaction(tx_hash)
            
            if receipt and receipt['status'] == 1:
                contract_address = receipt['contractAddress']
                logger.info(f"合约部署成功: {contract_address}")
                return contract_address
            else:
                raise Exception("合约部署失败")
        
        except Exception as e:
            logger.error(f"部署合约失败: {e}")
            raise
    
    def load_contract(self, address: str, abi: List[Dict[str, Any]]) -> Any:
        """加载已部署的合约"""
        try:
            contract = self.w3.eth.contract(
                address=to_checksum_address(address),
                abi=abi
            )
            
            self.contracts[address] = contract
            return contract
        
        except Exception as e:
            logger.error(f"加载合约失败: {e}")
            raise
    
    def call_contract_function(self, contract_address: str, 
                             function_name: str,
                             args: List[Any] = None,
                             wallet: Optional[WalletInfo] = None,
                             password: Optional[str] = None) -> Any:
        """调用合约函数"""
        try:
            contract = self.contracts.get(contract_address)
            if not contract:
                raise ValueError(f"合约未加载: {contract_address}")
            
            function = getattr(contract.functions, function_name)
            
            if args:
                function = function(*args)
            else:
                function = function()
            
            # 如果是只读函数，直接调用
            if not wallet:
                return function.call()
            
            # 如果是写入函数，需要发送交易
            if wallet.is_encrypted and password:
                private_key = self._decrypt_private_key(wallet, password)
            else:
                private_key = wallet.private_key
            
            account = Account.from_key(private_key)
            
            # 估算Gas
            gas_estimate = function.estimateGas({'from': account.address})
            
            # 构建交易
            transaction = function.buildTransaction({
                'from': account.address,
                'gas': gas_estimate,
                'gasPrice': self.provider.get_gas_price(),
                'nonce': self.provider.get_nonce(account.address)
            })
            
            # 签名并发送交易
            signed_txn = account.sign_transaction(transaction)
            tx_hash = self.provider.send_transaction(signed_txn.rawTransaction)
            
            return tx_hash
        
        except Exception as e:
            logger.error(f"调用合约函数失败: {e}")
            raise
    
    def get_contract_events(self, contract_address: str, 
                          event_name: str,
                          from_block: int = 0,
                          to_block: str = 'latest') -> List[Dict[str, Any]]:
        """获取合约事件"""
        try:
            contract = self.contracts.get(contract_address)
            if not contract:
                raise ValueError(f"合约未加载: {contract_address}")
            
            event_filter = getattr(contract.events, event_name).createFilter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            events = event_filter.get_all_entries()
            return [dict(event) for event in events]
        
        except Exception as e:
            logger.error(f"获取合约事件失败: {e}")
            return []
    
    def _decrypt_private_key(self, wallet: WalletInfo, password: str) -> str:
        """解密私钥（简化版本）"""
        # 这里应该使用WalletManager的解密方法
        return wallet.private_key


class TokenManager:
    """代币管理器"""
    
    def __init__(self, web3_provider: Web3Provider, contract_manager: SmartContractManager):
        self.provider = web3_provider
        self.contract_manager = contract_manager
        self.tokens: Dict[str, TokenInfo] = {}
        
        # ERC20 ABI（简化版本）
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "transfer",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            }
        ]
    
    def add_token(self, address: str) -> Optional[TokenInfo]:
        """添加代币"""
        try:
            contract = self.contract_manager.load_contract(address, self.erc20_abi)
            
            # 获取代币信息
            name = contract.functions.name().call()
            symbol = contract.functions.symbol().call()
            decimals = contract.functions.decimals().call()
            
            token_info = TokenInfo(
                address=to_checksum_address(address),
                symbol=symbol,
                name=name,
                decimals=decimals,
                standard=TokenStandard.ERC20
            )
            
            self.tokens[address] = token_info
            logger.info(f"代币已添加: {symbol} ({address})")
            
            return token_info
        
        except Exception as e:
            logger.error(f"添加代币失败: {e}")
            return None
    
    def get_token_balance(self, token_address: str, wallet_address: str) -> Decimal:
        """获取代币余额"""
        try:
            contract = self.contract_manager.contracts.get(token_address)
            if not contract:
                # 尝试加载合约
                contract = self.contract_manager.load_contract(token_address, self.erc20_abi)
            
            balance = contract.functions.balanceOf(to_checksum_address(wallet_address)).call()
            
            # 获取代币精度
            token_info = self.tokens.get(token_address)
            decimals = token_info.decimals if token_info else 18
            
            return Decimal(balance) / Decimal(10 ** decimals)
        
        except Exception as e:
            logger.error(f"获取代币余额失败: {e}")
            return Decimal('0')
    
    def transfer_token(self, token_address: str, to_address: str, amount: Decimal,
                      wallet: WalletInfo, password: Optional[str] = None) -> str:
        """转账代币"""
        try:
            # 获取代币信息
            token_info = self.tokens.get(token_address)
            if not token_info:
                raise ValueError(f"未知代币: {token_address}")
            
            # 转换金额为最小单位
            amount_wei = int(amount * Decimal(10 ** token_info.decimals))
            
            # 调用合约转账函数
            tx_hash = self.contract_manager.call_contract_function(
                token_address,
                'transfer',
                [to_checksum_address(to_address), amount_wei],
                wallet,
                password
            )
            
            logger.info(f"代币转账已发送: {tx_hash}")
            return tx_hash
        
        except Exception as e:
            logger.error(f"代币转账失败: {e}")
            raise
    
    def get_token_info(self, address: str) -> Optional[TokenInfo]:
        """获取代币信息"""
        return self.tokens.get(to_checksum_address(address))
    
    def list_tokens(self) -> List[TokenInfo]:
        """列出所有代币"""
        return list(self.tokens.values())


class NFTManager:
    """NFT管理器"""
    
    def __init__(self, web3_provider: Web3Provider, contract_manager: SmartContractManager):
        self.provider = web3_provider
        self.contract_manager = contract_manager
        self.nfts: Dict[str, List[NFTInfo]] = {}  # contract_address -> [NFTInfo]
        
        # ERC721 ABI（简化版本）
        self.erc721_abi = [
            {
                "constant": True,
                "inputs": [{"name": "tokenId", "type": "uint256"}],
                "name": "ownerOf",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "tokenId", "type": "uint256"}],
                "name": "tokenURI",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "from", "type": "address"},
                    {"name": "to", "type": "address"},
                    {"name": "tokenId", "type": "uint256"}
                ],
                "name": "transferFrom",
                "outputs": [],
                "type": "function"
            }
        ]
    
    async def get_nft_metadata(self, token_uri: str) -> Optional[NFTMetadata]:
        """获取NFT元数据"""
        try:
            async with aiohttp.ClientSession() as session:
                # 处理IPFS URI
                if token_uri.startswith('ipfs://'):
                    token_uri = token_uri.replace('ipfs://', 'https://ipfs.io/ipfs/')
                
                async with session.get(token_uri) as response:
                    if response.status == 200:
                        metadata = await response.json()
                        
                        return NFTMetadata(
                            name=metadata.get('name', ''),
                            description=metadata.get('description', ''),
                            image=metadata.get('image', ''),
                            external_url=metadata.get('external_url'),
                            attributes=metadata.get('attributes', []),
                            background_color=metadata.get('background_color'),
                            animation_url=metadata.get('animation_url'),
                            youtube_url=metadata.get('youtube_url')
                        )
        
        except Exception as e:
            logger.error(f"获取NFT元数据失败: {e}")
            return None
    
    async def get_nft_info(self, contract_address: str, token_id: int) -> Optional[NFTInfo]:
        """获取NFT信息"""
        try:
            contract = self.contract_manager.contracts.get(contract_address)
            if not contract:
                contract = self.contract_manager.load_contract(contract_address, self.erc721_abi)
            
            # 获取所有者
            owner = contract.functions.ownerOf(token_id).call()
            
            # 获取token URI
            token_uri = contract.functions.tokenURI(token_id).call()
            
            # 获取元数据
            metadata = await self.get_nft_metadata(token_uri)
            
            if not metadata:
                metadata = NFTMetadata(
                    name=f"Token #{token_id}",
                    description="",
                    image=""
                )
            
            nft_info = NFTInfo(
                contract_address=to_checksum_address(contract_address),
                token_id=token_id,
                owner=owner,
                metadata=metadata,
                token_uri=token_uri,
                standard=TokenStandard.ERC721
            )
            
            return nft_info
        
        except Exception as e:
            logger.error(f"获取NFT信息失败: {e}")
            return None
    
    def transfer_nft(self, contract_address: str, token_id: int, to_address: str,
                    wallet: WalletInfo, password: Optional[str] = None) -> str:
        """转移NFT"""
        try:
            tx_hash = self.contract_manager.call_contract_function(
                contract_address,
                'transferFrom',
                [wallet.address, to_checksum_address(to_address), token_id],
                wallet,
                password
            )
            
            logger.info(f"NFT转移已发送: {tx_hash}")
            return tx_hash
        
        except Exception as e:
            logger.error(f"NFT转移失败: {e}")
            raise


class DeFiManager:
    """DeFi协议管理器"""
    
    def __init__(self, web3_provider: Web3Provider, contract_manager: SmartContractManager):
        self.provider = web3_provider
        self.contract_manager = contract_manager
        self.protocols: Dict[str, Dict[str, Any]] = {}
    
    def add_protocol(self, name: str, contract_address: str, abi: List[Dict[str, Any]]):
        """添加DeFi协议"""
        try:
            contract = self.contract_manager.load_contract(contract_address, abi)
            
            self.protocols[name] = {
                'address': contract_address,
                'contract': contract,
                'abi': abi
            }
            
            logger.info(f"DeFi协议已添加: {name}")
        
        except Exception as e:
            logger.error(f"添加DeFi协议失败: {e}")
            raise
    
    def swap_tokens(self, protocol_name: str, token_in: str, token_out: str,
                   amount_in: Decimal, min_amount_out: Decimal,
                   wallet: WalletInfo, password: Optional[str] = None) -> str:
        """代币交换"""
        try:
            protocol = self.protocols.get(protocol_name)
            if not protocol:
                raise ValueError(f"未知协议: {protocol_name}")
            
            # 这里应该根据具体的DeFi协议实现交换逻辑
            # 以Uniswap V2为例
            tx_hash = self.contract_manager.call_contract_function(
                protocol['address'],
                'swapExactTokensForTokens',
                [
                    int(amount_in * Decimal(10**18)),  # 假设18位精度
                    int(min_amount_out * Decimal(10**18)),
                    [token_in, token_out],
                    wallet.address,
                    int(time.time()) + 300  # 5分钟过期
                ],
                wallet,
                password
            )
            
            logger.info(f"代币交换已发送: {tx_hash}")
            return tx_hash
        
        except Exception as e:
            logger.error(f"代币交换失败: {e}")
            raise
    
    def add_liquidity(self, protocol_name: str, token_a: str, token_b: str,
                     amount_a: Decimal, amount_b: Decimal,
                     wallet: WalletInfo, password: Optional[str] = None) -> str:
        """添加流动性"""
        try:
            protocol = self.protocols.get(protocol_name)
            if not protocol:
                raise ValueError(f"未知协议: {protocol_name}")
            
            tx_hash = self.contract_manager.call_contract_function(
                protocol['address'],
                'addLiquidity',
                [
                    token_a,
                    token_b,
                    int(amount_a * Decimal(10**18)),
                    int(amount_b * Decimal(10**18)),
                    0,  # min_amount_a
                    0,  # min_amount_b
                    wallet.address,
                    int(time.time()) + 300
                ],
                wallet,
                password
            )
            
            logger.info(f"添加流动性已发送: {tx_hash}")
            return tx_hash
        
        except Exception as e:
            logger.error(f"添加流动性失败: {e}")
            raise
    
    def stake_tokens(self, protocol_name: str, amount: Decimal,
                    wallet: WalletInfo, password: Optional[str] = None) -> str:
        """质押代币"""
        try:
            protocol = self.protocols.get(protocol_name)
            if not protocol:
                raise ValueError(f"未知协议: {protocol_name}")
            
            tx_hash = self.contract_manager.call_contract_function(
                protocol['address'],
                'stake',
                [int(amount * Decimal(10**18))],
                wallet,
                password
            )
            
            logger.info(f"代币质押已发送: {tx_hash}")
            return tx_hash
        
        except Exception as e:
            logger.error(f"代币质押失败: {e}")
            raise


class Web3Manager:
    """Web3集成管理器"""
    
    def __init__(self, network_config: NetworkConfig):
        self.network_config = network_config
        self.provider = Web3Provider(network_config)
        self.wallet_manager = WalletManager()
        self.contract_manager = SmartContractManager(self.provider)
        self.token_manager = TokenManager(self.provider, self.contract_manager)
        self.nft_manager = NFTManager(self.provider, self.contract_manager)
        self.defi_manager = DeFiManager(self.provider, self.contract_manager)
        
        logger.info(f"Web3管理器已初始化: {network_config.name}")
    
    def create_wallet(self, name: Optional[str] = None, password: Optional[str] = None) -> WalletInfo:
        """创建钱包"""
        return self.wallet_manager.create_wallet(name, password)
    
    def import_wallet(self, private_key: Optional[str] = None, 
                     mnemonic: Optional[str] = None,
                     name: Optional[str] = None,
                     password: Optional[str] = None) -> WalletInfo:
        """导入钱包"""
        if private_key:
            return self.wallet_manager.import_wallet_from_private_key(private_key, name, password)
        elif mnemonic:
            return self.wallet_manager.import_wallet_from_mnemonic(mnemonic, name=name, password=password)
        else:
            raise ValueError("必须提供私钥或助记词")
    
    def send_transaction(self, from_wallet: WalletInfo, to_address: str, 
                        amount: Decimal, password: Optional[str] = None) -> str:
        """发送交易"""
        try:
            # 获取私钥
            if from_wallet.is_encrypted and password:
                private_key = self.wallet_manager.decrypt_private_key(from_wallet, password)
            else:
                private_key = from_wallet.private_key
            
            account = Account.from_key(private_key)
            
            # 构建交易
            transaction = {
                'to': to_checksum_address(to_address),
                'value': self.provider.w3.toWei(amount, 'ether'),
                'gas': 21000,
                'gasPrice': self.provider.get_gas_price(),
                'nonce': self.provider.get_nonce(account.address)
            }
            
            # 签名交易
            signed_txn = account.sign_transaction(transaction)
            
            # 发送交易
            tx_hash = self.provider.send_transaction(signed_txn.rawTransaction)
            
            logger.info(f"交易已发送: {tx_hash}")
            return tx_hash
        
        except Exception as e:
            logger.error(f"发送交易失败: {e}")
            raise
    
    def get_balance(self, address: str) -> Decimal:
        """获取余额"""
        return self.provider.get_balance(address)
    
    def get_transaction_status(self, tx_hash: str) -> Optional[TransactionInfo]:
        """获取交易状态"""
        return self.provider.get_transaction(tx_hash)
    
    async def monitor_address(self, address: str, callback: Callable[[TransactionInfo], None]):
        """监控地址交易"""
        # 这里应该实现实时监控逻辑
        # 可以使用WebSocket或定期轮询
        pass


# 示例使用
async def example_usage():
    """示例用法"""
    # 网络配置
    network_config = NetworkConfig(
        name="Ethereum Goerli",
        network=BlockchainNetwork.ETHEREUM_GOERLI,
        rpc_url="https://goerli.infura.io/v3/YOUR_PROJECT_ID",
        chain_id=5,
        currency_symbol="ETH",
        block_explorer_url="https://goerli.etherscan.io",
        is_testnet=True
    )
    
    # 创建Web3管理器
    web3_manager = Web3Manager(network_config)
    
    # 创建钱包
    wallet = web3_manager.create_wallet("My Wallet", "password123")
    print(f"钱包已创建: {wallet.address}")
    
    # 获取余额
    balance = web3_manager.get_balance(wallet.address)
    print(f"钱包余额: {balance} ETH")
    
    # 添加ERC20代币
    token_address = "0x..."  # 代币合约地址
    token_info = web3_manager.token_manager.add_token(token_address)
    if token_info:
        print(f"代币已添加: {token_info.symbol}")
        
        # 获取代币余额
        token_balance = web3_manager.token_manager.get_token_balance(
            token_address, wallet.address
        )
        print(f"代币余额: {token_balance} {token_info.symbol}")
    
    # 获取NFT信息
    nft_contract = "0x..."  # NFT合约地址
    nft_token_id = 1
    nft_info = await web3_manager.nft_manager.get_nft_info(nft_contract, nft_token_id)
    if nft_info:
        print(f"NFT: {nft_info.metadata.name}")
        print(f"所有者: {nft_info.owner}")
    
    # 智能合约示例
    contract_source = """
    pragma solidity ^0.8.0;
    
    contract SimpleStorage {
        uint256 public storedData;
        
        constructor(uint256 _initialValue) {
            storedData = _initialValue;
        }
        
        function set(uint256 _value) public {
            storedData = _value;
        }
        
        function get() public view returns (uint256) {
            return storedData;
        }
    }
    """
    
    try:
        # 编译合约
        contract_info = web3_manager.contract_manager.compile_contract(
            contract_source, "SimpleStorage"
        )
        
        # 部署合约
        contract_address = web3_manager.contract_manager.deploy_contract(
            contract_info, [42], wallet, "password123"
        )
        
        print(f"合约已部署: {contract_address}")
        
        # 加载合约
        contract = web3_manager.contract_manager.load_contract(
            contract_address, contract_info['abi']
        )
        
        # 调用只读函数
        stored_value = web3_manager.contract_manager.call_contract_function(
            contract_address, "get"
        )
        print(f"存储的值: {stored_value}")
        
        # 调用写入函数
        tx_hash = web3_manager.contract_manager.call_contract_function(
            contract_address, "set", [100], wallet, "password123"
        )
        print(f"设置值交易: {tx_hash}")
    
    except Exception as e:
        print(f"合约操作失败: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())