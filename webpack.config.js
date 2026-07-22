const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin');


module.exports = {
    mode: 'development',
    entry: './src/application/index.ts',
    output: {
        path: path.resolve(__dirname, 'public'),
        filename: 'bundle.js'
    },
    devtool: 'inline-source-map',
    devServer: {
        static: [
            { directory: path.resolve(__dirname), watch: false },
        ],
        open: true,
        port: 9001,
        proxy: [
            {
                context: [
                    '/predict_rotation',
                    '/save_annotations',
                    '/connect-to-workstation',
                ],
                target: 'http://127.0.0.1:5000',
            },
        ]
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: './src/application/index.html'
        })
    ],
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/
            }, 
            {
                test: /\.css$/,
                use: [{
                    loader: "style-loader"
                }, {
                    loader: "css-loader"
                }]
            },
            {
                test: /\.(png|jpe?g|gif|jp2|webp)$/,
                type: 'asset/resource',
                generator: {
                    filename: '[name][ext]',
                },
            },
            {
                test: /\.txt$/,
                type: 'asset/source',
            }
        ]
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js']
    },

}
