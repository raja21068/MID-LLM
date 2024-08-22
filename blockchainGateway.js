const { Gateway, Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');

async function interactWithBlockchain(operation, dataId, data) {
    const ccpPath = path.resolve(__dirname, 'connection.json');
    const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));
    const walletPath = path.join(process.cwd(), 'wallet');
    const wallet = await Wallets.newFileSystemWallet(walletPath);
    const gateway = new Gateway();

    try {
        await gateway.connect(ccp, { wallet, identity: 'appUser', discovery: { enabled: true, asLocalhost: true } });
        const network = await gateway.getNetwork('mychannel');
        const contract = network.getContract('mycc');

        if (operation === 'submit') {
            await contract.submitTransaction('StoreMedicalData', dataId, data);
            console.log('Transaction has been submitted');
        } else if (operation === 'query') {
            const result = await contract.evaluateTransaction('QueryMedicalData', dataId);
            console.log(`Query result: ${result.toString()}`);
        }
    } finally {
        gateway.disconnect();
    }
}
